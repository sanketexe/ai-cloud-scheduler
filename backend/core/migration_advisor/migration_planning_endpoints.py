"""
API Endpoints for Migration Planning Engine

This module provides REST API endpoints for migration plan generation,
phase management, and progress tracking.

Requirements: 4.1, 4.4
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import structlog

from ..database import get_db_session
from ..auth import get_current_user
from ..models import User
from .models import (
    MigrationProject, MigrationPlan, MigrationPhase,
    PhaseStatus, MigrationRiskLevel
)
from .migration_planning_engine import MigrationPlanningEngine

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/migrations", tags=["Migration Planning"])


# Request/Response Models

class GenerateMigrationPlanRequest(BaseModel):
    """Request to generate a migration plan"""
    target_provider: str = Field(..., description="Target cloud provider (aws, gcp, azure)")
    workload_ids: Optional[List[str]] = Field(None, description="List of workload IDs to migrate")
    migration_strategy: str = Field(default="phased", description="Migration strategy (phased, big_bang, parallel)")
    target_start_date: Optional[str] = Field(None, description="Target migration start date (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "target_provider": "aws",
                "workload_ids": ["workload-1", "workload-2"],
                "migration_strategy": "phased",
                "target_start_date": "2024-01-15T00:00:00"
            }
        }


class UpdatePhaseStatusRequest(BaseModel):
    """Request to update migration phase status"""
    status: str = Field(..., description="New phase status")
    notes: Optional[str] = Field(None, description="Status update notes")
    actual_start_date: Optional[str] = Field(None, description="Actual start date (ISO format)")
    actual_end_date: Optional[str] = Field(None, description="Actual end date (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "in_progress",
                "notes": "Phase started successfully",
                "actual_start_date": "2024-01-15T09:00:00"
            }
        }


class MigrationPhaseResponse(BaseModel):
    """Response with migration phase details"""
    phase_id: str
    phase_name: str
    phase_order: int
    workloads: List[str]
    start_date: Optional[str]
    end_date: Optional[str]
    actual_start_date: Optional[str]
    actual_end_date: Optional[str]
    status: str
    prerequisites: List[str]
    success_criteria: List[str]
    notes: Optional[str]


class MigrationPlanResponse(BaseModel):
    """Response with complete migration plan"""
    plan_id: str
    target_provider: str
    total_duration_days: int
    estimated_cost: float
    risk_level: str
    phases: List[MigrationPhaseResponse]
    dependencies_graph: Dict[str, Any]
    migration_waves: List[Dict[str, Any]]
    success_criteria: List[str]
    created_at: str


class MigrationProgressResponse(BaseModel):
    """Response with migration progress tracking"""
    plan_id: str
    overall_status: str
    total_phases: int
    completed_phases: int
    in_progress_phases: int
    not_started_phases: int
    failed_phases: int
    overall_progress_percentage: float
    current_phase: Optional[MigrationPhaseResponse]
    next_phase: Optional[MigrationPhaseResponse]
    estimated_completion_date: Optional[str]
    actual_start_date: Optional[str]
    days_elapsed: Optional[int]
    days_remaining: Optional[int]


# API Endpoints

@router.post("/{project_id}/plan", response_model=MigrationPlanResponse, status_code=status.HTTP_201_CREATED)
async def generate_migration_plan(
    project_id: str,
    request: GenerateMigrationPlanRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Generate a comprehensive migration plan for the project.
    
    This endpoint creates a detailed migration plan with phases, timelines,
    dependencies, and cost estimates based on the selected cloud provider
    and workload requirements.
    
    Requirements: 4.1
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
        
        # Initialize planning engine
        engine = MigrationPlanningEngine(db)
        
        # Parse target start date if provided
        target_start = None
        if request.target_start_date:
            target_start = datetime.fromisoformat(request.target_start_date.replace('Z', '+00:00'))
        
        # Generate migration plan
        plan = engine.generate_migration_plan(
            project_id=project_id,
            target_provider=request.target_provider,
            workload_ids=request.workload_ids,
            migration_strategy=request.migration_strategy,
            target_start_date=target_start
        )
        
        db.commit()
        
        logger.info(
            "Migration plan generated",
            project_id=project_id,
            plan_id=plan.plan_id,
            target_provider=request.target_provider,
            total_phases=len(plan.phases)
        )
        
        # Convert to response
        return _convert_plan_to_response(plan)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to generate migration plan", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate migration plan: {str(e)}"
        )


@router.get("/{project_id}/plan", response_model=MigrationPlanResponse)
async def get_migration_plan(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Retrieve the migration plan for a project.
    
    Returns the complete migration plan including all phases, timelines,
    dependencies, and cost estimates.
    
    Requirements: 4.1
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
        
        # Get migration plan
        plan = db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
        
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No migration plan found for project {project_id}"
            )
        
        return _convert_plan_to_response(plan)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve migration plan", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve migration plan: {str(e)}"
        )


@router.put("/{project_id}/plan/phases/{phase_id}/status")
async def update_phase_status(
    project_id: str,
    phase_id: str,
    request: UpdatePhaseStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Update the status of a migration phase.
    
    This endpoint allows tracking phase progress by updating status,
    recording actual start/end dates, and adding notes.
    
    Requirements: 4.4
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
        
        # Get migration plan
        plan = db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
        
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No migration plan found for project {project_id}"
            )
        
        # Get phase
        phase = db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == plan.id,
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Phase {phase_id} not found"
            )
        
        # Validate status
        try:
            new_status = PhaseStatus(request.status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[e.value for e in PhaseStatus]}"
            )
        
        # Update phase
        phase.status = new_status
        if request.notes:
            phase.notes = request.notes
        if request.actual_start_date:
            phase.actual_start_date = datetime.fromisoformat(request.actual_start_date.replace('Z', '+00:00'))
        if request.actual_end_date:
            phase.actual_end_date = datetime.fromisoformat(request.actual_end_date.replace('Z', '+00:00'))
        
        db.commit()
        
        logger.info(
            "Phase status updated",
            project_id=project_id,
            phase_id=phase_id,
            new_status=new_status.value
        )
        
        return {
            'phase_id': phase.phase_id,
            'phase_name': phase.phase_name,
            'status': phase.status.value,
            'actual_start_date': phase.actual_start_date.isoformat() if phase.actual_start_date else None,
            'actual_end_date': phase.actual_end_date.isoformat() if phase.actual_end_date else None,
            'notes': phase.notes,
            'updated_at': phase.updated_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to update phase status", error=str(e), project_id=project_id, phase_id=phase_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update phase status: {str(e)}"
        )


@router.get("/{project_id}/plan/progress", response_model=MigrationProgressResponse)
async def get_migration_progress(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get migration progress tracking information.
    
    Returns comprehensive progress metrics including phase completion status,
    overall progress percentage, current phase, and timeline estimates.
    
    Requirements: 4.4
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
        
        # Get migration plan
        plan = db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
        
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No migration plan found for project {project_id}"
            )
        
        # Calculate progress metrics
        phases = plan.phases
        total_phases = len(phases)
        
        completed_phases = sum(1 for p in phases if p.status == PhaseStatus.COMPLETED)
        in_progress_phases = sum(1 for p in phases if p.status == PhaseStatus.IN_PROGRESS)
        not_started_phases = sum(1 for p in phases if p.status == PhaseStatus.NOT_STARTED)
        failed_phases = sum(1 for p in phases if p.status == PhaseStatus.FAILED)
        
        overall_progress = (completed_phases / total_phases * 100) if total_phases > 0 else 0
        
        # Find current and next phases
        current_phase = next((p for p in phases if p.status == PhaseStatus.IN_PROGRESS), None)
        next_phase = None
        if not current_phase:
            next_phase = next((p for p in phases if p.status == PhaseStatus.NOT_STARTED), None)
        
        # Calculate timeline metrics
        actual_start = min((p.actual_start_date for p in phases if p.actual_start_date), default=None)
        days_elapsed = None
        days_remaining = None
        estimated_completion = None
        
        if actual_start:
            days_elapsed = (datetime.utcnow() - actual_start).days
            if plan.total_duration_days:
                days_remaining = max(0, plan.total_duration_days - days_elapsed)
                estimated_completion = actual_start.replace(tzinfo=None) + timedelta(days=plan.total_duration_days)
        
        # Determine overall status
        if failed_phases > 0:
            overall_status = "failed"
        elif completed_phases == total_phases:
            overall_status = "completed"
        elif in_progress_phases > 0:
            overall_status = "in_progress"
        else:
            overall_status = "not_started"
        
        return MigrationProgressResponse(
            plan_id=plan.plan_id,
            overall_status=overall_status,
            total_phases=total_phases,
            completed_phases=completed_phases,
            in_progress_phases=in_progress_phases,
            not_started_phases=not_started_phases,
            failed_phases=failed_phases,
            overall_progress_percentage=overall_progress,
            current_phase=_convert_phase_to_response(current_phase) if current_phase else None,
            next_phase=_convert_phase_to_response(next_phase) if next_phase else None,
            estimated_completion_date=estimated_completion.isoformat() if estimated_completion else None,
            actual_start_date=actual_start.isoformat() if actual_start else None,
            days_elapsed=days_elapsed,
            days_remaining=days_remaining
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get migration progress", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get migration progress: {str(e)}"
        )


# Helper Functions

def _convert_plan_to_response(plan: MigrationPlan) -> MigrationPlanResponse:
    """Convert migration plan to response model"""
    return MigrationPlanResponse(
        plan_id=plan.plan_id,
        target_provider=plan.target_provider,
        total_duration_days=plan.total_duration_days,
        estimated_cost=float(plan.estimated_cost),
        risk_level=plan.risk_level.value,
        phases=[_convert_phase_to_response(phase) for phase in sorted(plan.phases, key=lambda p: p.phase_order)],
        dependencies_graph=plan.dependencies_graph or {},
        migration_waves=plan.migration_waves or [],
        success_criteria=plan.success_criteria or [],
        created_at=plan.created_at.isoformat()
    )


def _convert_phase_to_response(phase: MigrationPhase) -> MigrationPhaseResponse:
    """Convert migration phase to response model"""
    return MigrationPhaseResponse(
        phase_id=phase.phase_id,
        phase_name=phase.phase_name,
        phase_order=phase.phase_order,
        workloads=phase.workloads or [],
        start_date=phase.start_date.isoformat() if phase.start_date else None,
        end_date=phase.end_date.isoformat() if phase.end_date else None,
        actual_start_date=phase.actual_start_date.isoformat() if phase.actual_start_date else None,
        actual_end_date=phase.actual_end_date.isoformat() if phase.actual_end_date else None,
        status=phase.status.value,
        prerequisites=phase.prerequisites or [],
        success_criteria=phase.success_criteria or [],
        notes=phase.notes
    )


# Add missing import
from datetime import timedelta
