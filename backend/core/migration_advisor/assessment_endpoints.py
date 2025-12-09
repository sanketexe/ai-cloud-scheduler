"""
API endpoints for Migration Assessment Engine

This module provides REST API endpoints for migration project management,
organization profiling, and assessment workflows.
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from ..database import get_db_session
from ..auth import get_current_user
from ..models import User
from .assessment_engine import MigrationAssessmentEngine
from .models import (
    MigrationStatus, CompanySize, InfrastructureType, ExperienceLevel
)

router = APIRouter(prefix="/api/migrations", tags=["Migration Assessment"])


# Request/Response Models

class InitiateMigrationRequest(BaseModel):
    """Request to initiate a new migration assessment"""
    organization_name: str = Field(..., min_length=1, max_length=255)
    
    class Config:
        schema_extra = {
            "example": {
                "organization_name": "Acme Corporation"
            }
        }


class OrganizationProfileRequest(BaseModel):
    """Request to create/update organization profile"""
    company_size: str = Field(..., description="Company size: small, medium, large, enterprise")
    industry: str = Field(..., min_length=1, max_length=100)
    current_infrastructure: str = Field(..., description="Infrastructure type: on_premises, cloud, hybrid, multi_cloud")
    it_team_size: int = Field(..., ge=0, description="Number of IT team members")
    cloud_experience_level: str = Field(..., description="Experience level: none, beginner, intermediate, advanced")
    geographic_presence: Optional[List[str]] = Field(default=[], description="List of regions/countries")
    additional_context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context information")
    
    @validator('company_size')
    def validate_company_size(cls, v):
        try:
            CompanySize(v)
        except ValueError:
            raise ValueError(f"Invalid company_size. Must be one of: {[e.value for e in CompanySize]}")
        return v
    
    @validator('current_infrastructure')
    def validate_infrastructure(cls, v):
        try:
            InfrastructureType(v)
        except ValueError:
            raise ValueError(f"Invalid infrastructure type. Must be one of: {[e.value for e in InfrastructureType]}")
        return v
    
    @validator('cloud_experience_level')
    def validate_experience(cls, v):
        try:
            ExperienceLevel(v)
        except ValueError:
            raise ValueError(f"Invalid experience level. Must be one of: {[e.value for e in ExperienceLevel]}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "company_size": "medium",
                "industry": "Financial Services",
                "current_infrastructure": "on_premises",
                "it_team_size": 15,
                "cloud_experience_level": "beginner",
                "geographic_presence": ["North America", "Europe"],
                "additional_context": {
                    "compliance_requirements": ["SOC2", "GDPR"]
                }
            }
        }


class UpdateProjectStatusRequest(BaseModel):
    """Request to update project status"""
    status: str = Field(..., description="New migration status")
    current_phase: Optional[str] = Field(None, description="Current phase description")
    
    @validator('status')
    def validate_status(cls, v):
        try:
            MigrationStatus(v)
        except ValueError:
            raise ValueError(f"Invalid status. Must be one of: {[e.value for e in MigrationStatus]}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "status": "analysis",
                "current_phase": "Workload Analysis"
            }
        }


class MigrationProjectResponse(BaseModel):
    """Response with migration project details"""
    project_id: str
    project_uuid: str
    organization_name: str
    status: str
    current_phase: Optional[str]
    estimated_completion: Optional[str]
    created_at: str
    
    class Config:
        schema_extra = {
            "example": {
                "project_id": "mig-acme-corporation-20231116120000-abc123de",
                "project_uuid": "550e8400-e29b-41d4-a716-446655440000",
                "organization_name": "Acme Corporation",
                "status": "assessment",
                "current_phase": "Initial Assessment",
                "estimated_completion": "2023-12-01T00:00:00",
                "created_at": "2023-11-16T12:00:00"
            }
        }


class OrganizationProfileResponse(BaseModel):
    """Response with organization profile and timeline"""
    profile: Dict[str, Any]
    timeline_estimation: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "profile": {
                    "company_size": "medium",
                    "industry": "Financial Services",
                    "current_infrastructure": "on_premises",
                    "it_team_size": 15,
                    "cloud_experience_level": "beginner",
                    "geographic_presence": ["North America", "Europe"]
                },
                "timeline_estimation": {
                    "estimated_days": 18,
                    "estimated_completion_date": "2023-12-04T00:00:00",
                    "breakdown": {
                        "base_days": 14,
                        "infrastructure_multiplier": 1.0,
                        "experience_adjustment_days": 3,
                        "team_size_factor": 1.0
                    }
                }
            }
        }


class ProjectListResponse(BaseModel):
    """Response with list of projects"""
    projects: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class ValidationResponse(BaseModel):
    """Response with validation results"""
    is_complete: bool
    missing_items: List[str]
    warnings: List[str]


# API Endpoints

@router.post("/projects", response_model=MigrationProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_migration_project(
    request: InitiateMigrationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Create a new migration project and initiate assessment.
    
    This endpoint creates a migration project with ASSESSMENT status and returns
    the project details including a unique project_id.
    """
    try:
        engine = MigrationAssessmentEngine(db)
        result = engine.initiate_migration_assessment(
            organization_name=request.organization_name,
            created_by_user_id=current_user.id
        )
        
        return MigrationProjectResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create migration project: {str(e)}"
        )


@router.get("/projects/{project_id}", response_model=MigrationProjectResponse)
async def get_migration_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Retrieve a migration project by project_id.
    
    Returns detailed information about the migration project including status,
    current phase, and estimated completion.
    """
    try:
        engine = MigrationAssessmentEngine(db)
        project = engine.project_manager.get_project(project_id)
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        return MigrationProjectResponse(
            project_id=project.project_id,
            project_uuid=str(project.id),
            organization_name=project.organization_name,
            status=project.status.value,
            current_phase=project.current_phase,
            estimated_completion=project.estimated_completion.isoformat() if project.estimated_completion else None,
            created_at=project.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve project: {str(e)}"
        )


@router.get("/projects", response_model=ProjectListResponse)
async def list_migration_projects(
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List migration projects with optional filtering.
    
    Query parameters:
    - status_filter: Filter by migration status (optional)
    - limit: Maximum number of results (default: 100)
    - offset: Pagination offset (default: 0)
    """
    try:
        engine = MigrationAssessmentEngine(db)
        
        # Parse status filter if provided
        status_enum = None
        if status_filter:
            try:
                status_enum = MigrationStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status filter. Must be one of: {[e.value for e in MigrationStatus]}"
                )
        
        projects = engine.project_manager.list_projects(
            status=status_enum,
            created_by=current_user.id,
            limit=limit,
            offset=offset
        )
        
        projects_data = [
            {
                'project_id': p.project_id,
                'project_uuid': str(p.id),
                'organization_name': p.organization_name,
                'status': p.status.value,
                'current_phase': p.current_phase,
                'estimated_completion': p.estimated_completion.isoformat() if p.estimated_completion else None,
                'created_at': p.created_at.isoformat()
            }
            for p in projects
        ]
        
        return ProjectListResponse(
            projects=projects_data,
            total=len(projects_data),
            limit=limit,
            offset=offset
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list projects: {str(e)}"
        )


@router.put("/projects/{project_id}/status")
async def update_project_status(
    project_id: str,
    request: UpdateProjectStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Update migration project status and current phase.
    
    This endpoint allows transitioning the project through different migration phases.
    Invalid status transitions will be rejected.
    """
    try:
        engine = MigrationAssessmentEngine(db)
        
        status_enum = MigrationStatus(request.status)
        project = engine.project_manager.update_project_status(
            project_id=project_id,
            new_status=status_enum,
            current_phase=request.current_phase
        )
        
        db.commit()
        
        return {
            'project_id': project.project_id,
            'status': project.status.value,
            'current_phase': project.current_phase,
            'updated_at': project.updated_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update project status: {str(e)}"
        )


@router.post("/{project_id}/assessment/organization", response_model=OrganizationProfileResponse)
async def create_organization_profile(
    project_id: str,
    request: OrganizationProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Create organization profile and estimate assessment timeline.
    
    This endpoint collects comprehensive organization information and automatically
    estimates the assessment timeline based on company size, infrastructure complexity,
    and team experience.
    """
    try:
        engine = MigrationAssessmentEngine(db)
        
        # Verify project exists and user has access
        project = engine.project_manager.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Create profile and get timeline estimation
        result = engine.collect_organization_profile(
            project_id=project_id,
            profile_data=request.dict()
        )
        
        db.commit()
        
        return OrganizationProfileResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create organization profile: {str(e)}"
        )


@router.get("/{project_id}/assessment/organization")
async def get_organization_profile(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Retrieve organization profile for a migration project.
    """
    try:
        engine = MigrationAssessmentEngine(db)
        
        # Get project
        project = engine.project_manager.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Get profile
        profile = engine.profiler.get_profile(project.id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Organization profile not found for project {project_id}"
            )
        
        return {
            'company_size': profile.company_size.value,
            'industry': profile.industry,
            'current_infrastructure': profile.current_infrastructure.value,
            'it_team_size': profile.it_team_size,
            'cloud_experience_level': profile.cloud_experience_level.value,
            'geographic_presence': profile.geographic_presence,
            'additional_context': profile.additional_context,
            'created_at': profile.created_at.isoformat(),
            'updated_at': profile.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve organization profile: {str(e)}"
        )


@router.get("/{project_id}/assessment/status", response_model=ValidationResponse)
async def get_assessment_status(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Check assessment completeness and validation status.
    
    Returns information about which assessment components have been completed
    and what is still missing.
    """
    try:
        engine = MigrationAssessmentEngine(db)
        
        # Verify project exists
        project = engine.project_manager.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        validation = engine.validate_assessment_completeness(project_id)
        
        return ValidationResponse(**validation)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate assessment: {str(e)}"
        )


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_migration_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Delete (cancel) a migration project.
    
    This performs a soft delete by setting the project status to CANCELLED.
    """
    try:
        engine = MigrationAssessmentEngine(db)
        
        success = engine.project_manager.delete_project(project_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project: {str(e)}"
        )
