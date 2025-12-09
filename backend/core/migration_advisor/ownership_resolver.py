"""
Ownership Resolver for Cloud Migration Advisor

This module identifies unassigned resources and suggests ownership based on
naming patterns, tags, and relationships.

Requirements: 5.5
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .resource_discovery_engine import CloudResource, ResourceInventory
from .auto_categorization_engine import ResourceCategorization, CategorizedResources
from .organizational_structure_manager import OrganizationalStructure, Team, Project


logger = logging.getLogger(__name__)


class OwnershipStatus(Enum):
    """Status of resource ownership"""
    ASSIGNED = "assigned"
    UNASSIGNED = "unassigned"
    PENDING = "pending"
    SUGGESTED = "suggested"


@dataclass
class OwnershipSuggestion:
    """Suggestion for resource ownership"""
    resource_id: str
    suggested_team: Optional[str] = None
    suggested_project: Optional[str] = None
    confidence_score: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    alternative_suggestions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UnassignedResource:
    """Represents an unassigned resource"""
    resource: CloudResource
    categorization: Optional[ResourceCategorization] = None
    missing_dimensions: List[str] = field(default_factory=list)
    
    def is_fully_unassigned(self) -> bool:
        """Check if resource has no assignments at all"""
        if not self.categorization:
            return True
        
        return not any([
            self.categorization.team,
            self.categorization.project,
            self.categorization.environment,
            self.categorization.region
        ])


@dataclass
class OwnershipAssignment:
    """Represents an ownership assignment"""
    resource_id: str
    team: Optional[str] = None
    project: Optional[str] = None
    assigned_by: str = "system"  # system, user, suggested
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    notes: Optional[str] = None


@dataclass
class OwnershipResolutionResult:
    """Result of ownership resolution process"""
    total_resources: int
    assigned_count: int = 0
    unassigned_count: int = 0
    suggestions: List[OwnershipSuggestion] = field(default_factory=list)
    assignments: List[OwnershipAssignment] = field(default_factory=list)


class OwnershipResolver:
    """
    Resolves resource ownership by identifying unassigned resources
    and suggesting ownership based on patterns and relationships
    
    Requirements: 5.5
    """
    
    def __init__(self):
        """Initialize the ownership resolver"""
        logger.info("Ownership Resolver initialized")
    
    def identify_unassigned_resources(
        self,
        inventory: ResourceInventory,
        categorizations: CategorizedResources
    ) -> List[UnassignedResource]:
        """
        Identify resources that lack ownership assignments
        
        Args:
            inventory: Resource inventory
            categorizations: Resource categorizations
            
        Returns:
            List of unassigned resources
        """
        logger.info(f"Identifying unassigned resources from {inventory.total_count} resources")
        
        unassigned = []
        
        for resource in inventory.resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            
            # Check if resource is missing critical dimensions
            missing_dimensions = []
            
            if not categorization:
                # Completely uncategorized
                unassigned.append(UnassignedResource(
                    resource=resource,
                    categorization=None,
                    missing_dimensions=["team", "project", "environment", "region"]
                ))
                continue
            
            # Check for missing dimensions
            if not categorization.team:
                missing_dimensions.append("team")
            if not categorization.project:
                missing_dimensions.append("project")
            if not categorization.environment:
                missing_dimensions.append("environment")
            if not categorization.region:
                missing_dimensions.append("region")
            
            # If any critical dimensions are missing, mark as unassigned
            if missing_dimensions:
                unassigned.append(UnassignedResource(
                    resource=resource,
                    categorization=categorization,
                    missing_dimensions=missing_dimensions
                ))
        
        logger.info(f"Found {len(unassigned)} unassigned resources")
        
        return unassigned
    
    def suggest_ownership(
        self,
        unassigned_resources: List[UnassignedResource],
        structure: OrganizationalStructure,
        all_categorizations: CategorizedResources
    ) -> List[OwnershipSuggestion]:
        """
        Suggest ownership for unassigned resources
        
        Args:
            unassigned_resources: List of unassigned resources
            structure: Organizational structure
            all_categorizations: All resource categorizations for context
            
        Returns:
            List of ownership suggestions
        """
        logger.info(f"Generating ownership suggestions for {len(unassigned_resources)} resources")
        
        suggestions = []
        
        for unassigned in unassigned_resources:
            suggestion = self._generate_suggestion(
                unassigned,
                structure,
                all_categorizations
            )
            suggestions.append(suggestion)
        
        logger.info(f"Generated {len(suggestions)} ownership suggestions")
        
        return suggestions
    
    def _generate_suggestion(
        self,
        unassigned: UnassignedResource,
        structure: OrganizationalStructure,
        all_categorizations: CategorizedResources
    ) -> OwnershipSuggestion:
        """
        Generate ownership suggestion for a single resource
        
        Args:
            unassigned: Unassigned resource
            structure: Organizational structure
            all_categorizations: All categorizations for context
            
        Returns:
            OwnershipSuggestion
        """
        suggestion = OwnershipSuggestion(resource_id=unassigned.resource.resource_id)
        
        # Try multiple suggestion strategies
        strategies = [
            self._suggest_from_naming_pattern,
            self._suggest_from_tags,
            self._suggest_from_relationships,
            self._suggest_from_resource_type,
        ]
        
        for strategy in strategies:
            team, project, confidence, reasoning = strategy(
                unassigned,
                structure,
                all_categorizations
            )
            
            if team or project:
                # Found a suggestion
                if confidence > suggestion.confidence_score:
                    # This is a better suggestion
                    if suggestion.suggested_team or suggestion.suggested_project:
                        # Save previous suggestion as alternative
                        suggestion.alternative_suggestions.append({
                            "team": suggestion.suggested_team,
                            "project": suggestion.suggested_project,
                            "confidence": suggestion.confidence_score,
                            "reasoning": suggestion.reasoning
                        })
                    
                    suggestion.suggested_team = team
                    suggestion.suggested_project = project
                    suggestion.confidence_score = confidence
                    suggestion.reasoning = reasoning
                else:
                    # Add as alternative suggestion
                    suggestion.alternative_suggestions.append({
                        "team": team,
                        "project": project,
                        "confidence": confidence,
                        "reasoning": reasoning
                    })
        
        return suggestion
    
    def _suggest_from_naming_pattern(
        self,
        unassigned: UnassignedResource,
        structure: OrganizationalStructure,
        all_categorizations: CategorizedResources
    ) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """
        Suggest ownership based on resource naming patterns
        
        Returns:
            Tuple of (team, project, confidence, reasoning)
        """
        resource = unassigned.resource
        resource_name = resource.resource_name.lower()
        
        # Check if resource name contains team names
        for team in structure.teams:
            team_name_lower = team.name.lower()
            if team_name_lower in resource_name:
                # Found team name in resource name
                reasoning = [f"Resource name contains team name '{team.name}'"]
                
                # Try to find a project for this team
                project = None
                for proj in structure.projects:
                    if proj.owner_team_id == team.team_id:
                        project = proj.name
                        reasoning.append(f"Project '{project}' is owned by team '{team.name}'")
                        break
                
                confidence = 0.7  # Moderate confidence from naming pattern
                return team.name, project, confidence, reasoning
        
        # Check if resource name contains project names
        for project in structure.projects:
            project_name_lower = project.name.lower()
            if project_name_lower in resource_name:
                # Found project name in resource name
                reasoning = [f"Resource name contains project name '{project.name}'"]
                
                # Get team that owns this project
                team = None
                if project.owner_team_id:
                    team_obj = structure.get_team_by_id(project.owner_team_id)
                    if team_obj:
                        team = team_obj.name
                        reasoning.append(f"Team '{team}' owns project '{project.name}'")
                
                confidence = 0.7  # Moderate confidence from naming pattern
                return team, project.name, confidence, reasoning
        
        return None, None, 0.0, []
    
    def _suggest_from_tags(
        self,
        unassigned: UnassignedResource,
        structure: OrganizationalStructure,
        all_categorizations: CategorizedResources
    ) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """
        Suggest ownership based on resource tags
        
        Returns:
            Tuple of (team, project, confidence, reasoning)
        """
        resource = unassigned.resource
        tags = resource.tags
        
        reasoning = []
        team = None
        project = None
        
        # Check for owner tag
        if "owner" in tags:
            owner = tags["owner"]
            reasoning.append(f"Resource has owner tag: {owner}")
            
            # Try to match owner to a team
            for team_obj in structure.teams:
                if owner.lower() in team_obj.name.lower() or owner in team_obj.members:
                    team = team_obj.name
                    reasoning.append(f"Owner matches team '{team}'")
                    break
        
        # Check for department tag
        if "department" in tags:
            dept = tags["department"]
            reasoning.append(f"Resource has department tag: {dept}")
            
            # Try to match department to a team
            for team_obj in structure.teams:
                if dept.lower() in team_obj.name.lower():
                    team = team_obj.name
                    reasoning.append(f"Department matches team '{team}'")
                    break
        
        # Check for application tag
        if "application" in tags or "app" in tags:
            app = tags.get("application") or tags.get("app")
            reasoning.append(f"Resource has application tag: {app}")
            
            # Try to match application to a project
            for proj in structure.projects:
                if app.lower() in proj.name.lower():
                    project = proj.name
                    reasoning.append(f"Application matches project '{project}'")
                    
                    # Get team for this project
                    if proj.owner_team_id and not team:
                        team_obj = structure.get_team_by_id(proj.owner_team_id)
                        if team_obj:
                            team = team_obj.name
                            reasoning.append(f"Team '{team}' owns project '{project}'")
                    break
        
        if team or project:
            confidence = 0.8  # High confidence from tags
            return team, project, confidence, reasoning
        
        return None, None, 0.0, []
    
    def _suggest_from_relationships(
        self,
        unassigned: UnassignedResource,
        structure: OrganizationalStructure,
        all_categorizations: CategorizedResources
    ) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """
        Suggest ownership based on relationships with other resources
        
        Returns:
            Tuple of (team, project, confidence, reasoning)
        """
        resource = unassigned.resource
        reasoning = []
        
        # Check if resource has parent resource
        if "parent_resource_id" in resource.metadata:
            parent_id = resource.metadata["parent_resource_id"]
            parent_cat = all_categorizations.get_categorization(parent_id)
            
            if parent_cat and (parent_cat.team or parent_cat.project):
                reasoning.append(f"Resource has parent resource: {parent_id}")
                
                if parent_cat.team:
                    reasoning.append(f"Parent resource is owned by team '{parent_cat.team}'")
                if parent_cat.project:
                    reasoning.append(f"Parent resource belongs to project '{parent_cat.project}'")
                
                confidence = 0.9  # High confidence from parent relationship
                return parent_cat.team, parent_cat.project, confidence, reasoning
        
        # Check if resource is in same VPC/network as other resources
        if "vpc_id" in resource.metadata or "network_id" in resource.metadata:
            network_id = resource.metadata.get("vpc_id") or resource.metadata.get("network_id")
            
            # Find other resources in same network
            same_network_resources = []
            for cat in all_categorizations.categorizations:
                # This would require access to all resources, simplified for now
                pass
            
            # If majority of resources in same network belong to a team/project,
            # suggest that ownership
        
        return None, None, 0.0, []
    
    def _suggest_from_resource_type(
        self,
        unassigned: UnassignedResource,
        structure: OrganizationalStructure,
        all_categorizations: CategorizedResources
    ) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """
        Suggest ownership based on resource type patterns
        
        Returns:
            Tuple of (team, project, confidence, reasoning)
        """
        resource = unassigned.resource
        reasoning = []
        
        # Look for patterns in how teams use certain resource types
        # For example, if a team primarily uses databases, suggest database resources to them
        
        # Count resource types by team
        team_resource_types: Dict[str, Dict[str, int]] = {}
        
        for cat in all_categorizations.categorizations:
            if cat.team:
                if cat.team not in team_resource_types:
                    team_resource_types[cat.team] = {}
                
                # This would require access to resource type, simplified for now
                # resource_type = get_resource_type(cat.resource_id)
                # if resource_type not in team_resource_types[cat.team]:
                #     team_resource_types[cat.team][resource_type] = 0
                # team_resource_types[cat.team][resource_type] += 1
        
        # For now, return no suggestion from this strategy
        return None, None, 0.0, []
    
    def assign_ownership(
        self,
        resource_id: str,
        team: Optional[str],
        project: Optional[str],
        assigned_by: str = "user",
        notes: Optional[str] = None
    ) -> OwnershipAssignment:
        """
        Create an ownership assignment
        
        Args:
            resource_id: Resource to assign
            team: Team to assign to
            project: Project to assign to
            assigned_by: Who made the assignment
            notes: Optional notes
            
        Returns:
            OwnershipAssignment
        """
        logger.info(f"Assigning ownership for resource {resource_id}")
        
        assignment = OwnershipAssignment(
            resource_id=resource_id,
            team=team,
            project=project,
            assigned_by=assigned_by,
            notes=notes
        )
        
        return assignment
    
    def resolve_ownership(
        self,
        inventory: ResourceInventory,
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        auto_assign_high_confidence: bool = False,
        confidence_threshold: float = 0.8
    ) -> OwnershipResolutionResult:
        """
        Complete ownership resolution workflow
        
        Args:
            inventory: Resource inventory
            categorizations: Resource categorizations
            structure: Organizational structure
            auto_assign_high_confidence: Whether to auto-assign high confidence suggestions
            confidence_threshold: Minimum confidence for auto-assignment
            
        Returns:
            OwnershipResolutionResult
        """
        logger.info("Starting ownership resolution workflow")
        
        result = OwnershipResolutionResult(total_resources=inventory.total_count)
        
        # Identify unassigned resources
        unassigned = self.identify_unassigned_resources(inventory, categorizations)
        result.unassigned_count = len(unassigned)
        result.assigned_count = inventory.total_count - len(unassigned)
        
        # Generate suggestions
        suggestions = self.suggest_ownership(unassigned, structure, categorizations)
        result.suggestions = suggestions
        
        # Auto-assign high confidence suggestions if enabled
        if auto_assign_high_confidence:
            for suggestion in suggestions:
                if suggestion.confidence_score >= confidence_threshold:
                    assignment = self.assign_ownership(
                        resource_id=suggestion.resource_id,
                        team=suggestion.suggested_team,
                        project=suggestion.suggested_project,
                        assigned_by="system_auto",
                        notes=f"Auto-assigned with confidence {suggestion.confidence_score:.2f}"
                    )
                    result.assignments.append(assignment)
        
        logger.info(
            f"Ownership resolution complete: {result.assigned_count} assigned, "
            f"{result.unassigned_count} unassigned, {len(result.suggestions)} suggestions, "
            f"{len(result.assignments)} auto-assignments"
        )
        
        return result
    
    def get_ownership_summary(
        self,
        result: OwnershipResolutionResult
    ) -> Dict[str, Any]:
        """
        Get summary of ownership resolution
        
        Args:
            result: Ownership resolution result
            
        Returns:
            Dictionary with summary statistics
        """
        high_confidence_suggestions = sum(
            1 for s in result.suggestions if s.confidence_score >= 0.8
        )
        
        medium_confidence_suggestions = sum(
            1 for s in result.suggestions if 0.5 <= s.confidence_score < 0.8
        )
        
        low_confidence_suggestions = sum(
            1 for s in result.suggestions if s.confidence_score < 0.5
        )
        
        return {
            "total_resources": result.total_resources,
            "assigned_count": result.assigned_count,
            "unassigned_count": result.unassigned_count,
            "assignment_rate": result.assigned_count / result.total_resources if result.total_resources > 0 else 0,
            "suggestions_count": len(result.suggestions),
            "auto_assignments_count": len(result.assignments),
            "high_confidence_suggestions": high_confidence_suggestions,
            "medium_confidence_suggestions": medium_confidence_suggestions,
            "low_confidence_suggestions": low_confidence_suggestions
        }
