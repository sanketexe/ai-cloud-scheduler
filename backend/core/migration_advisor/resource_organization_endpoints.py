"""
API Endpoints for Resource Organization Engine

This module provides REST API endpoints for resource discovery, organization,
categorization, and inventory management.

Requirements: 5.1, 5.2
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import structlog

from ..database import get_db_session
from ..auth import get_current_user
from ..models import User
from .models import (
    MigrationProject, CategorizedResource, OrganizationalStructure,
    OwnershipStatus
)
from .resource_organization_engine import ResourceOrganizationEngine

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/migrations", tags=["Resource Organization"])


# Request/Response Models

class ResourceDiscoveryRequest(BaseModel):
    """Request to discover cloud resources"""
    provider: str = Field(..., description="Cloud provider (aws, gcp, azure)")
    credentials: Dict[str, str] = Field(..., description="Provider credentials")
    regions: Optional[List[str]] = Field(None, description="Regions to scan")
    resource_types: Optional[List[str]] = Field(None, description="Resource types to discover")
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "aws",
                "credentials": {
                    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
                    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
                },
                "regions": ["us-east-1", "us-west-2"],
                "resource_types": ["ec2", "s3", "rds"]
            }
        }


class OrganizeResourcesRequest(BaseModel):
    """Request to organize discovered resources"""
    organizational_structure_id: Optional[str] = Field(None, description="ID of organizational structure to use")
    auto_categorize: bool = Field(default=True, description="Automatically categorize resources")
    apply_tags: bool = Field(default=True, description="Apply tags to resources")
    
    class Config:
        schema_extra = {
            "example": {
                "organizational_structure_id": "org-struct-123",
                "auto_categorize": True,
                "apply_tags": True
            }
        }


class CategorizeResourceRequest(BaseModel):
    """Request to categorize a specific resource"""
    team: Optional[str] = Field(None, description="Team assignment")
    project: Optional[str] = Field(None, description="Project assignment")
    environment: Optional[str] = Field(None, description="Environment (dev, staging, prod)")
    region: Optional[str] = Field(None, description="Region assignment")
    cost_center: Optional[str] = Field(None, description="Cost center assignment")
    custom_attributes: Optional[Dict[str, str]] = Field(None, description="Custom attributes")
    tags: Optional[Dict[str, str]] = Field(None, description="Resource tags")
    
    class Config:
        schema_extra = {
            "example": {
                "team": "platform-engineering",
                "project": "web-application",
                "environment": "production",
                "region": "us-east-1",
                "cost_center": "engineering",
                "custom_attributes": {
                    "owner": "john.doe@example.com"
                },
                "tags": {
                    "Team": "platform-engineering",
                    "Environment": "production"
                }
            }
        }


class ResourceInventoryResponse(BaseModel):
    """Response with resource inventory"""
    total_resources: int
    resources_by_type: Dict[str, int]
    resources_by_status: Dict[str, int]
    resources: List[Dict[str, Any]]


class ResourceDiscoveryResponse(BaseModel):
    """Response from resource discovery"""
    discovery_id: str
    provider: str
    total_discovered: int
    resources_by_type: Dict[str, int]
    regions_scanned: List[str]
    discovery_timestamp: str


class OrganizationResultResponse(BaseModel):
    """Response from resource organization"""
    total_organized: int
    auto_categorized: int
    manually_categorized: int
    unassigned: int
    tags_applied: int


# API Endpoints

@router.post("/{project_id}/resources/discover", response_model=ResourceDiscoveryResponse)
async def discover_resources(
    project_id: str,
    request: ResourceDiscoveryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Discover cloud resources from the specified provider.
    
    This endpoint connects to the cloud provider and discovers all resources
    in the specified regions. The discovered resources are stored in the
    inventory for subsequent organization and categorization.
    
    Requirements: 5.1
    """
    try:
        # Verify project exists
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Initialize resource organization engine
        engine = ResourceOrganizationEngine(db)
        
        # Discover resources
        result = engine.discover_resources(
            project_id=project_id,
            provider=request.provider,
            credentials=request.credentials,
            regions=request.regions,
            resource_types=request.resource_types
        )
        
        db.commit()
        
        logger.info(
            "Resources discovered",
            project_id=project_id,
            provider=request.provider,
            total_discovered=result['total_discovered']
        )
        
        return ResourceDiscoveryResponse(
            discovery_id=result['discovery_id'],
            provider=result['provider'],
            total_discovered=result['total_discovered'],
            resources_by_type=result['resources_by_type'],
            regions_scanned=result['regions_scanned'],
            discovery_timestamp=result['discovery_timestamp']
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to discover resources", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to discover resources: {str(e)}"
        )


@router.post("/{project_id}/resources/organize", response_model=OrganizationResultResponse)
async def organize_resources(
    project_id: str,
    request: OrganizeResourcesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Organize discovered resources using organizational structure.
    
    This endpoint applies organizational structure to discovered resources,
    automatically categorizing them by team, project, environment, etc.,
    and optionally applying tags.
    
    Requirements: 5.2
    """
    try:
        # Verify project exists
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Initialize resource organization engine
        engine = ResourceOrganizationEngine(db)
        
        # Organize resources
        result = engine.organize_resources(
            project_id=project_id,
            organizational_structure_id=request.organizational_structure_id,
            auto_categorize=request.auto_categorize,
            apply_tags=request.apply_tags
        )
        
        db.commit()
        
        logger.info(
            "Resources organized",
            project_id=project_id,
            total_organized=result['total_organized'],
            auto_categorized=result['auto_categorized']
        )
        
        return OrganizationResultResponse(
            total_organized=result['total_organized'],
            auto_categorized=result['auto_categorized'],
            manually_categorized=result['manually_categorized'],
            unassigned=result['unassigned'],
            tags_applied=result['tags_applied']
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to organize resources", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to organize resources: {str(e)}"
        )


@router.get("/{project_id}/resources", response_model=ResourceInventoryResponse)
async def get_resource_inventory(
    project_id: str,
    resource_type: Optional[str] = None,
    ownership_status: Optional[str] = None,
    team: Optional[str] = None,
    project_filter: Optional[str] = None,
    environment: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get resource inventory with optional filtering.
    
    Returns a list of all discovered and categorized resources with
    optional filtering by type, ownership status, team, project, etc.
    
    Requirements: 5.1, 5.2
    """
    try:
        # Verify project exists
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Build query
        query = db.query(CategorizedResource).filter(
            CategorizedResource.migration_project_id == project.id
        )
        
        # Apply filters
        if resource_type:
            query = query.filter(CategorizedResource.resource_type == resource_type)
        
        if ownership_status:
            try:
                status_enum = OwnershipStatus(ownership_status)
                query = query.filter(CategorizedResource.ownership_status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid ownership status. Must be one of: {[e.value for e in OwnershipStatus]}"
                )
        
        if team:
            query = query.filter(CategorizedResource.team == team)
        
        if project_filter:
            query = query.filter(CategorizedResource.project == project_filter)
        
        if environment:
            query = query.filter(CategorizedResource.environment == environment)
        
        # Get total count
        total_resources = query.count()
        
        # Apply pagination
        resources = query.limit(limit).offset(offset).all()
        
        # Calculate aggregations
        all_resources = query.all()
        resources_by_type = {}
        resources_by_status = {}
        
        for resource in all_resources:
            resources_by_type[resource.resource_type] = resources_by_type.get(resource.resource_type, 0) + 1
            resources_by_status[resource.ownership_status.value] = resources_by_status.get(resource.ownership_status.value, 0) + 1
        
        # Convert to response
        resources_data = [
            {
                'id': str(resource.id),
                'resource_id': resource.resource_id,
                'resource_type': resource.resource_type,
                'resource_name': resource.resource_name,
                'provider': resource.provider,
                'team': resource.team,
                'project': resource.project,
                'environment': resource.environment,
                'region': resource.region,
                'cost_center': resource.cost_center,
                'custom_attributes': resource.custom_attributes,
                'tags': resource.tags,
                'ownership_status': resource.ownership_status.value,
                'created_at': resource.created_at.isoformat(),
                'updated_at': resource.updated_at.isoformat()
            }
            for resource in resources
        ]
        
        return ResourceInventoryResponse(
            total_resources=total_resources,
            resources_by_type=resources_by_type,
            resources_by_status=resources_by_status,
            resources=resources_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get resource inventory", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource inventory: {str(e)}"
        )


@router.put("/{project_id}/resources/{resource_id}/categorize")
async def categorize_resource(
    project_id: str,
    resource_id: str,
    request: CategorizeResourceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Manually categorize a specific resource.
    
    This endpoint allows manual assignment of organizational attributes
    to a resource, overriding any automatic categorization.
    
    Requirements: 5.2
    """
    try:
        # Verify project exists
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Get resource
        resource = db.query(CategorizedResource).filter(
            CategorizedResource.migration_project_id == project.id,
            CategorizedResource.resource_id == resource_id
        ).first()
        
        if not resource:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resource {resource_id} not found"
            )
        
        # Update categorization
        if request.team is not None:
            resource.team = request.team
        if request.project is not None:
            resource.project = request.project
        if request.environment is not None:
            resource.environment = request.environment
        if request.region is not None:
            resource.region = request.region
        if request.cost_center is not None:
            resource.cost_center = request.cost_center
        if request.custom_attributes is not None:
            resource.custom_attributes = request.custom_attributes
        if request.tags is not None:
            resource.tags = request.tags
        
        # Update ownership status
        if any([request.team, request.project, request.environment]):
            resource.ownership_status = OwnershipStatus.ASSIGNED
        
        db.commit()
        
        logger.info(
            "Resource categorized",
            project_id=project_id,
            resource_id=resource_id,
            team=request.team,
            project=request.project
        )
        
        return {
            'resource_id': resource.resource_id,
            'resource_type': resource.resource_type,
            'team': resource.team,
            'project': resource.project,
            'environment': resource.environment,
            'region': resource.region,
            'cost_center': resource.cost_center,
            'ownership_status': resource.ownership_status.value,
            'updated_at': resource.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to categorize resource", error=str(e), project_id=project_id, resource_id=resource_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to categorize resource: {str(e)}"
        )
