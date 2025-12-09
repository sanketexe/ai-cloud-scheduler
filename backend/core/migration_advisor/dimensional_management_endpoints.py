"""
API Endpoints for Multi-Dimensional Resource Management

This module provides REST API endpoints for managing organizational dimensions,
generating dimensional views, and advanced resource filtering.

Requirements: 6.1, 6.2, 6.3
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import structlog

from ..database import get_db_session
from ..auth import get_current_user
from ..models import User
from .models import OrganizationalStructure, CategorizedResource
from .dimensional_view_engine import DimensionalViewEngine
from .advanced_filtering_system import AdvancedFilteringSystem

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Dimensional Management"])


# Request/Response Models

class CreateDimensionRequest(BaseModel):
    """Request to create an organizational dimension"""
    dimension_type: str = Field(..., description="Type of dimension (team, project, environment, region, cost_center)")
    dimension_name: str = Field(..., description="Name of the dimension")
    dimension_values: List[str] = Field(..., description="Possible values for this dimension")
    parent_dimension: Optional[str] = Field(None, description="Parent dimension for hierarchical structures")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "dimension_type": "team",
                "dimension_name": "Engineering Teams",
                "dimension_values": ["platform", "backend", "frontend", "mobile"],
                "metadata": {
                    "description": "Engineering team structure"
                }
            }
        }


class DimensionalViewRequest(BaseModel):
    """Request to generate a dimensional view"""
    dimension_type: str = Field(..., description="Dimension to view by")
    aggregation_metrics: Optional[List[str]] = Field(None, description="Metrics to aggregate (cost, count, etc.)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    
    class Config:
        schema_extra = {
            "example": {
                "dimension_type": "team",
                "aggregation_metrics": ["resource_count", "estimated_cost"],
                "filters": {
                    "environment": "production"
                }
            }
        }


class AdvancedFilterRequest(BaseModel):
    """Request for advanced resource filtering"""
    filter_expression: Dict[str, Any] = Field(..., description="Complex filter expression")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", description="Sort order (asc, desc)")
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "filter_expression": {
                    "operator": "AND",
                    "conditions": [
                        {
                            "field": "team",
                            "operator": "equals",
                            "value": "platform"
                        },
                        {
                            "field": "environment",
                            "operator": "in",
                            "value": ["production", "staging"]
                        }
                    ]
                },
                "sort_by": "resource_name",
                "sort_order": "asc",
                "limit": 50,
                "offset": 0
            }
        }


class DimensionResponse(BaseModel):
    """Response with dimension details"""
    dimension_id: str
    dimension_type: str
    dimension_name: str
    dimension_values: List[str]
    parent_dimension: Optional[str]
    metadata: Dict[str, Any]
    created_at: str


class DimensionalViewResponse(BaseModel):
    """Response with dimensional view data"""
    dimension_type: str
    view_data: List[Dict[str, Any]]
    aggregations: Dict[str, Any]
    total_resources: int


class FilteredResourcesResponse(BaseModel):
    """Response with filtered resources"""
    total_count: int
    filtered_count: int
    resources: List[Dict[str, Any]]
    filter_summary: Dict[str, Any]


# API Endpoints

@router.post("/organizations/dimensions", response_model=DimensionResponse, status_code=status.HTTP_201_CREATED)
async def create_dimension(
    request: CreateDimensionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Create a new organizational dimension.
    
    This endpoint creates a new dimension (team, project, environment, etc.)
    that can be used to organize and categorize resources.
    
    Requirements: 6.3
    """
    try:
        # Validate dimension type
        valid_types = ['team', 'project', 'environment', 'region', 'cost_center', 'custom']
        if request.dimension_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dimension type. Must be one of: {valid_types}"
            )
        
        # Create organizational structure entry
        # Note: In a full implementation, this would use a dedicated dimension management system
        org_structure = OrganizationalStructure(
            migration_project_id=None,  # Global dimension
            structure_name=request.dimension_name,
            teams=request.dimension_values if request.dimension_type == 'team' else [],
            projects=request.dimension_values if request.dimension_type == 'project' else [],
            environments=request.dimension_values if request.dimension_type == 'environment' else [],
            regions=request.dimension_values if request.dimension_type == 'region' else [],
            cost_centers=request.dimension_values if request.dimension_type == 'cost_center' else [],
            custom_dimensions={
                request.dimension_type: {
                    'name': request.dimension_name,
                    'values': request.dimension_values,
                    'parent': request.parent_dimension,
                    'metadata': request.metadata or {}
                }
            }
        )
        
        db.add(org_structure)
        db.commit()
        db.refresh(org_structure)
        
        logger.info(
            "Dimension created",
            dimension_type=request.dimension_type,
            dimension_name=request.dimension_name,
            values_count=len(request.dimension_values)
        )
        
        return DimensionResponse(
            dimension_id=str(org_structure.id),
            dimension_type=request.dimension_type,
            dimension_name=request.dimension_name,
            dimension_values=request.dimension_values,
            parent_dimension=request.parent_dimension,
            metadata=request.metadata or {},
            created_at=org_structure.created_at.isoformat()
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
        logger.error("Failed to create dimension", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dimension: {str(e)}"
        )


@router.get("/organizations/dimensions/{dimension_type}")
async def get_dimension(
    dimension_type: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Retrieve a specific organizational dimension.
    
    Returns the definition and values for the specified dimension type.
    
    Requirements: 6.3
    """
    try:
        # Query organizational structures for the dimension
        structures = db.query(OrganizationalStructure).all()
        
        # Aggregate dimension values
        dimension_values = set()
        dimension_name = dimension_type.replace('_', ' ').title()
        
        for structure in structures:
            if dimension_type == 'team':
                dimension_values.update(structure.teams or [])
            elif dimension_type == 'project':
                dimension_values.update(structure.projects or [])
            elif dimension_type == 'environment':
                dimension_values.update(structure.environments or [])
            elif dimension_type == 'region':
                dimension_values.update(structure.regions or [])
            elif dimension_type == 'cost_center':
                dimension_values.update(structure.cost_centers or [])
            elif dimension_type in (structure.custom_dimensions or {}):
                custom_dim = structure.custom_dimensions[dimension_type]
                dimension_values.update(custom_dim.get('values', []))
                dimension_name = custom_dim.get('name', dimension_name)
        
        if not dimension_values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dimension {dimension_type} not found or has no values"
            )
        
        return {
            'dimension_type': dimension_type,
            'dimension_name': dimension_name,
            'dimension_values': sorted(list(dimension_values)),
            'total_values': len(dimension_values)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dimension", error=str(e), dimension_type=dimension_type)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dimension: {str(e)}"
        )


@router.get("/resources/views/{dimension}", response_model=DimensionalViewResponse)
async def get_dimensional_view(
    dimension: str,
    project_id: Optional[str] = None,
    aggregation_metrics: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Generate a dimensional view of resources.
    
    This endpoint generates a view of resources organized by the specified
    dimension (team, project, environment, etc.) with aggregated metrics.
    
    Requirements: 6.1
    """
    try:
        # Build base query
        query = db.query(CategorizedResource)
        
        # Filter by project if specified
        if project_id:
            from .models import MigrationProject
            project = db.query(MigrationProject).filter(
                MigrationProject.project_id == project_id
            ).first()
            if project:
                query = query.filter(CategorizedResource.migration_project_id == project.id)
        
        resources = query.all()
        
        # Group by dimension
        dimension_groups = {}
        total_resources = len(resources)
        
        for resource in resources:
            dimension_value = None
            if dimension == 'team':
                dimension_value = resource.team or 'unassigned'
            elif dimension == 'project':
                dimension_value = resource.project or 'unassigned'
            elif dimension == 'environment':
                dimension_value = resource.environment or 'unassigned'
            elif dimension == 'region':
                dimension_value = resource.region or 'unassigned'
            elif dimension == 'cost_center':
                dimension_value = resource.cost_center or 'unassigned'
            else:
                dimension_value = 'unknown'
            
            if dimension_value not in dimension_groups:
                dimension_groups[dimension_value] = []
            dimension_groups[dimension_value].append(resource)
        
        # Build view data
        view_data = []
        for dim_value, dim_resources in dimension_groups.items():
            view_data.append({
                'dimension_value': dim_value,
                'resource_count': len(dim_resources),
                'resource_types': len(set(r.resource_type for r in dim_resources)),
                'resources': [
                    {
                        'resource_id': r.resource_id,
                        'resource_type': r.resource_type,
                        'resource_name': r.resource_name,
                        'provider': r.provider
                    }
                    for r in dim_resources[:10]  # Limit to first 10
                ]
            })
        
        # Sort by resource count
        view_data.sort(key=lambda x: x['resource_count'], reverse=True)
        
        # Calculate aggregations
        aggregations = {
            'total_dimension_values': len(dimension_groups),
            'average_resources_per_value': total_resources / len(dimension_groups) if dimension_groups else 0,
            'max_resources_in_value': max((len(resources) for resources in dimension_groups.values()), default=0),
            'min_resources_in_value': min((len(resources) for resources in dimension_groups.values()), default=0)
        }
        
        logger.info(
            "Dimensional view generated",
            dimension=dimension,
            total_resources=total_resources,
            dimension_values=len(dimension_groups)
        )
        
        return DimensionalViewResponse(
            dimension_type=dimension,
            view_data=view_data,
            aggregations=aggregations,
            total_resources=total_resources
        )
        
    except Exception as e:
        logger.error("Failed to generate dimensional view", error=str(e), dimension=dimension)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate dimensional view: {str(e)}"
        )


@router.post("/resources/filter", response_model=FilteredResourcesResponse)
async def filter_resources(
    request: AdvancedFilterRequest,
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Filter resources using advanced filter expressions.
    
    This endpoint supports complex filtering with multiple conditions,
    logical operators (AND, OR), and various comparison operators.
    
    Requirements: 6.2
    """
    try:
        # Build base query
        query = db.query(CategorizedResource)
        
        # Filter by project if specified
        if project_id:
            from .models import MigrationProject
            project = db.query(MigrationProject).filter(
                MigrationProject.project_id == project_id
            ).first()
            if project:
                query = query.filter(CategorizedResource.migration_project_id == project.id)
        
        total_count = query.count()
        
        # Apply advanced filters
        filtered_query = _apply_filter_expression(query, request.filter_expression)
        
        # Apply sorting
        if request.sort_by:
            sort_field = getattr(CategorizedResource, request.sort_by, None)
            if sort_field:
                if request.sort_order == 'desc':
                    filtered_query = filtered_query.order_by(sort_field.desc())
                else:
                    filtered_query = filtered_query.order_by(sort_field.asc())
        
        # Get filtered count
        filtered_count = filtered_query.count()
        
        # Apply pagination
        resources = filtered_query.limit(request.limit).offset(request.offset).all()
        
        # Convert to response
        resources_data = [
            {
                'id': str(r.id),
                'resource_id': r.resource_id,
                'resource_type': r.resource_type,
                'resource_name': r.resource_name,
                'provider': r.provider,
                'team': r.team,
                'project': r.project,
                'environment': r.environment,
                'region': r.region,
                'cost_center': r.cost_center,
                'ownership_status': r.ownership_status.value,
                'tags': r.tags
            }
            for r in resources
        ]
        
        # Build filter summary
        filter_summary = {
            'total_conditions': _count_conditions(request.filter_expression),
            'filter_efficiency': (filtered_count / total_count * 100) if total_count > 0 else 0,
            'resources_filtered_out': total_count - filtered_count
        }
        
        logger.info(
            "Resources filtered",
            total_count=total_count,
            filtered_count=filtered_count,
            filter_efficiency=filter_summary['filter_efficiency']
        )
        
        return FilteredResourcesResponse(
            total_count=total_count,
            filtered_count=filtered_count,
            resources=resources_data,
            filter_summary=filter_summary
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to filter resources", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to filter resources: {str(e)}"
        )


# Helper Functions

def _apply_filter_expression(query, filter_expr: Dict[str, Any]):
    """Apply filter expression to query"""
    operator = filter_expr.get('operator', 'AND')
    conditions = filter_expr.get('conditions', [])
    
    if not conditions:
        return query
    
    from sqlalchemy import and_, or_
    
    filters = []
    for condition in conditions:
        if 'operator' in condition and 'conditions' in condition:
            # Nested expression
            query = _apply_filter_expression(query, condition)
        else:
            # Simple condition
            field = condition.get('field')
            op = condition.get('operator')
            value = condition.get('value')
            
            if not field or not op:
                continue
            
            field_attr = getattr(CategorizedResource, field, None)
            if not field_attr:
                continue
            
            if op == 'equals':
                filters.append(field_attr == value)
            elif op == 'not_equals':
                filters.append(field_attr != value)
            elif op == 'in':
                filters.append(field_attr.in_(value))
            elif op == 'not_in':
                filters.append(~field_attr.in_(value))
            elif op == 'contains':
                filters.append(field_attr.contains(value))
            elif op == 'starts_with':
                filters.append(field_attr.startswith(value))
            elif op == 'ends_with':
                filters.append(field_attr.endswith(value))
    
    if filters:
        if operator == 'OR':
            query = query.filter(or_(*filters))
        else:
            query = query.filter(and_(*filters))
    
    return query


def _count_conditions(filter_expr: Dict[str, Any]) -> int:
    """Count total conditions in filter expression"""
    conditions = filter_expr.get('conditions', [])
    count = 0
    
    for condition in conditions:
        if 'operator' in condition and 'conditions' in condition:
            count += _count_conditions(condition)
        else:
            count += 1
    
    return count
