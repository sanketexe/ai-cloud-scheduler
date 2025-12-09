"""
Organizational Structure Manager for Cloud Migration Advisor

This module manages organizational structures including teams, projects, regions,
environments, and cost centers for resource categorization.

Requirements: 5.4
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class DimensionType(Enum):
    """Types of organizational dimensions"""
    TEAM = "team"
    PROJECT = "project"
    ENVIRONMENT = "environment"
    REGION = "region"
    COST_CENTER = "cost_center"
    DEPARTMENT = "department"
    CUSTOM = "custom"


@dataclass
class Dimension:
    """Represents an organizational dimension"""
    dimension_id: str
    dimension_type: DimensionType
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Ensure dimension_type is enum"""
        if isinstance(self.dimension_type, str):
            self.dimension_type = DimensionType(self.dimension_type)


@dataclass
class Team:
    """Represents a team in the organization"""
    team_id: str
    name: str
    description: Optional[str] = None
    lead: Optional[str] = None
    members: List[str] = field(default_factory=list)
    parent_team_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """Represents a project in the organization"""
    project_id: str
    name: str
    description: Optional[str] = None
    owner_team_id: Optional[str] = None
    budget: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Environment:
    """Represents an environment (dev, staging, prod, etc.)"""
    environment_id: str
    name: str
    description: Optional[str] = None
    environment_type: str = "production"  # development, staging, production, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Region:
    """Represents a geographic region"""
    region_id: str
    name: str
    cloud_provider_region: str
    country: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostCenter:
    """Represents a cost center for financial tracking"""
    cost_center_id: str
    name: str
    code: str
    budget: Optional[float] = None
    owner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrganizationalStructure:
    """
    Complete organizational structure definition
    
    Requirements: 5.4
    """
    structure_id: str
    name: str
    teams: List[Team] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    environments: List[Environment] = field(default_factory=list)
    regions: List[Region] = field(default_factory=list)
    cost_centers: List[CostCenter] = field(default_factory=list)
    custom_dimensions: Dict[str, List[Dimension]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_team_by_id(self, team_id: str) -> Optional[Team]:
        """Get team by ID"""
        for team in self.teams:
            if team.team_id == team_id:
                return team
        return None
    
    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        for project in self.projects:
            if project.project_id == project_id:
                return project
        return None
    
    def get_environment_by_id(self, environment_id: str) -> Optional[Environment]:
        """Get environment by ID"""
        for env in self.environments:
            if env.environment_id == environment_id:
                return env
        return None
    
    def get_region_by_id(self, region_id: str) -> Optional[Region]:
        """Get region by ID"""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None
    
    def get_cost_center_by_id(self, cost_center_id: str) -> Optional[CostCenter]:
        """Get cost center by ID"""
        for cc in self.cost_centers:
            if cc.cost_center_id == cost_center_id:
                return cc
        return None


@dataclass
class StructureValidationResult:
    """Result of structure validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class OrganizationalStructureManager:
    """
    Manages organizational structure creation, modification, and validation
    
    Requirements: 5.4
    """
    
    def __init__(self):
        """Initialize the organizational structure manager"""
        logger.info("Organizational Structure Manager initialized")
    
    def create_structure(
        self,
        structure_id: str,
        name: str
    ) -> OrganizationalStructure:
        """
        Create a new organizational structure
        
        Args:
            structure_id: Unique identifier for the structure
            name: Name of the organizational structure
            
        Returns:
            New OrganizationalStructure instance
        """
        logger.info(f"Creating organizational structure: {name}")
        
        structure = OrganizationalStructure(
            structure_id=structure_id,
            name=name
        )
        
        return structure
    
    def add_team(
        self,
        structure: OrganizationalStructure,
        team: Team
    ) -> OrganizationalStructure:
        """
        Add a team to the organizational structure
        
        Args:
            structure: Organizational structure to modify
            team: Team to add
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Adding team: {team.name}")
        
        # Check for duplicate team IDs
        if structure.get_team_by_id(team.team_id):
            raise ValueError(f"Team with ID {team.team_id} already exists")
        
        structure.teams.append(team)
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def add_project(
        self,
        structure: OrganizationalStructure,
        project: Project
    ) -> OrganizationalStructure:
        """
        Add a project to the organizational structure
        
        Args:
            structure: Organizational structure to modify
            project: Project to add
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Adding project: {project.name}")
        
        # Check for duplicate project IDs
        if structure.get_project_by_id(project.project_id):
            raise ValueError(f"Project with ID {project.project_id} already exists")
        
        structure.projects.append(project)
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def add_environment(
        self,
        structure: OrganizationalStructure,
        environment: Environment
    ) -> OrganizationalStructure:
        """
        Add an environment to the organizational structure
        
        Args:
            structure: Organizational structure to modify
            environment: Environment to add
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Adding environment: {environment.name}")
        
        # Check for duplicate environment IDs
        if structure.get_environment_by_id(environment.environment_id):
            raise ValueError(f"Environment with ID {environment.environment_id} already exists")
        
        structure.environments.append(environment)
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def add_region(
        self,
        structure: OrganizationalStructure,
        region: Region
    ) -> OrganizationalStructure:
        """
        Add a region to the organizational structure
        
        Args:
            structure: Organizational structure to modify
            region: Region to add
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Adding region: {region.name}")
        
        # Check for duplicate region IDs
        if structure.get_region_by_id(region.region_id):
            raise ValueError(f"Region with ID {region.region_id} already exists")
        
        structure.regions.append(region)
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def add_cost_center(
        self,
        structure: OrganizationalStructure,
        cost_center: CostCenter
    ) -> OrganizationalStructure:
        """
        Add a cost center to the organizational structure
        
        Args:
            structure: Organizational structure to modify
            cost_center: Cost center to add
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Adding cost center: {cost_center.name}")
        
        # Check for duplicate cost center IDs
        if structure.get_cost_center_by_id(cost_center.cost_center_id):
            raise ValueError(f"Cost center with ID {cost_center.cost_center_id} already exists")
        
        structure.cost_centers.append(cost_center)
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def add_custom_dimension(
        self,
        structure: OrganizationalStructure,
        dimension_category: str,
        dimension: Dimension
    ) -> OrganizationalStructure:
        """
        Add a custom dimension to the organizational structure
        
        Args:
            structure: Organizational structure to modify
            dimension_category: Category for the custom dimension
            dimension: Dimension to add
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Adding custom dimension: {dimension.name} to category {dimension_category}")
        
        if dimension_category not in structure.custom_dimensions:
            structure.custom_dimensions[dimension_category] = []
        
        structure.custom_dimensions[dimension_category].append(dimension)
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def validate_structure(
        self,
        structure: OrganizationalStructure
    ) -> StructureValidationResult:
        """
        Validate organizational structure for completeness and consistency
        
        Args:
            structure: Organizational structure to validate
            
        Returns:
            StructureValidationResult with validation status and messages
        """
        logger.info(f"Validating organizational structure: {structure.name}")
        
        errors = []
        warnings = []
        
        # Check if structure has at least one team
        if not structure.teams:
            warnings.append("No teams defined in organizational structure")
        
        # Check if structure has at least one project
        if not structure.projects:
            warnings.append("No projects defined in organizational structure")
        
        # Check if structure has environments
        if not structure.environments:
            warnings.append("No environments defined in organizational structure")
        
        # Check if structure has regions
        if not structure.regions:
            warnings.append("No regions defined in organizational structure")
        
        # Validate team references
        for project in structure.projects:
            if project.owner_team_id:
                if not structure.get_team_by_id(project.owner_team_id):
                    errors.append(
                        f"Project '{project.name}' references non-existent team ID: {project.owner_team_id}"
                    )
        
        # Validate parent team references
        for team in structure.teams:
            if team.parent_team_id:
                if not structure.get_team_by_id(team.parent_team_id):
                    errors.append(
                        f"Team '{team.name}' references non-existent parent team ID: {team.parent_team_id}"
                    )
        
        # Check for duplicate team names
        team_names = [team.name for team in structure.teams]
        if len(team_names) != len(set(team_names)):
            warnings.append("Duplicate team names found")
        
        # Check for duplicate project names
        project_names = [project.name for project in structure.projects]
        if len(project_names) != len(set(project_names)):
            warnings.append("Duplicate project names found")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Organizational structure validation passed")
        else:
            logger.warning(f"Organizational structure validation failed with {len(errors)} errors")
        
        return StructureValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def remove_team(
        self,
        structure: OrganizationalStructure,
        team_id: str
    ) -> OrganizationalStructure:
        """
        Remove a team from the organizational structure
        
        Args:
            structure: Organizational structure to modify
            team_id: ID of team to remove
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Removing team: {team_id}")
        
        structure.teams = [team for team in structure.teams if team.team_id != team_id]
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def remove_project(
        self,
        structure: OrganizationalStructure,
        project_id: str
    ) -> OrganizationalStructure:
        """
        Remove a project from the organizational structure
        
        Args:
            structure: Organizational structure to modify
            project_id: ID of project to remove
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Removing project: {project_id}")
        
        structure.projects = [proj for proj in structure.projects if proj.project_id != project_id]
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def update_team(
        self,
        structure: OrganizationalStructure,
        team_id: str,
        updates: Dict[str, Any]
    ) -> OrganizationalStructure:
        """
        Update a team in the organizational structure
        
        Args:
            structure: Organizational structure to modify
            team_id: ID of team to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated organizational structure
        """
        logger.debug(f"Updating team: {team_id}")
        
        team = structure.get_team_by_id(team_id)
        if not team:
            raise ValueError(f"Team with ID {team_id} not found")
        
        # Update allowed fields
        for key, value in updates.items():
            if hasattr(team, key):
                setattr(team, key, value)
        
        structure.updated_at = datetime.utcnow()
        
        return structure
    
    def get_structure_summary(
        self,
        structure: OrganizationalStructure
    ) -> Dict[str, Any]:
        """
        Get a summary of the organizational structure
        
        Args:
            structure: Organizational structure to summarize
            
        Returns:
            Dictionary with structure summary
        """
        return {
            "structure_id": structure.structure_id,
            "name": structure.name,
            "teams_count": len(structure.teams),
            "projects_count": len(structure.projects),
            "environments_count": len(structure.environments),
            "regions_count": len(structure.regions),
            "cost_centers_count": len(structure.cost_centers),
            "custom_dimensions_count": sum(len(dims) for dims in structure.custom_dimensions.values()),
            "created_at": structure.created_at.isoformat(),
            "updated_at": structure.updated_at.isoformat()
        }
