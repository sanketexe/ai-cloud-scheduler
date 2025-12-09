"""
API Endpoints for Post-Migration Integration

This module provides REST API endpoints for FinOps integration, baseline capture,
and migration report generation.

Requirements: 7.1, 7.2, 7.3, 7.5
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
    MigrationProject, BaselineMetrics, MigrationReport,
    OrganizationalStructure, CategorizedResource
)
from .post_migration_integration_engine import PostMigrationIntegrationEngine

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/migrations", tags=["Post-Migration Integration"])


# Request/Response Models

class FinOpsIntegrationRequest(BaseModel):
    """Request to integrate with FinOps platform"""
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    enable_budget_alerts: bool = Field(default=True, description="Enable budget alerts")
    enable_waste_detection: bool = Field(default=True, description="Enable waste detection")
    enable_optimization: bool = Field(default=True, description="Enable optimization recommendations")
    cost_allocation_method: str = Field(default="proportional", description="Cost allocation method")
    
    class Config:
        schema_extra = {
            "example": {
                "enable_cost_tracking": True,
                "enable_budget_alerts": True,
                "enable_waste_detection": True,
                "enable_optimization": True,
                "cost_allocation_method": "proportional"
            }
        }


class BaselineCaptureRequest(BaseModel):
    """Request to capture baseline metrics"""
    capture_cost_data: bool = Field(default=True, description="Capture cost data")
    capture_performance_data: bool = Field(default=True, description="Capture performance metrics")
    capture_utilization_data: bool = Field(default=True, description="Capture utilization metrics")
    baseline_period_days: int = Field(default=7, ge=1, le=30, description="Baseline period in days")
    
    class Config:
        schema_extra = {
            "example": {
                "capture_cost_data": True,
                "capture_performance_data": True,
                "capture_utilization_data": True,
                "baseline_period_days": 7
            }
        }


class FinOpsIntegrationResponse(BaseModel):
    """Response from FinOps integration"""
    integration_id: str
    status: str
    cost_tracking_enabled: bool
    budget_alerts_configured: int
    optimization_features_enabled: List[str]
    integration_timestamp: str


class BaselineMetricsResponse(BaseModel):
    """Response with baseline metrics"""
    baseline_id: str
    capture_date: str
    total_monthly_cost: float
    cost_by_service: Dict[str, float]
    cost_by_team: Dict[str, float]
    cost_by_project: Dict[str, float]
    cost_by_environment: Dict[str, float]
    resource_count: int
    resource_count_by_type: Dict[str, int]


class MigrationReportResponse(BaseModel):
    """Response with migration report"""
    report_id: str
    report_date: str
    start_date: str
    completion_date: str
    actual_duration_days: int
    planned_duration_days: int
    total_cost: float
    budgeted_cost: float
    cost_variance_percentage: float
    resources_migrated: int
    success_rate: float
    lessons_learned: List[str]
    optimization_opportunities: List[Dict[str, Any]]
    cost_analysis: Dict[str, Any]
    timeline_analysis: Dict[str, Any]


# API Endpoints

@router.post("/{project_id}/integration/finops", response_model=FinOpsIntegrationResponse)
async def integrate_with_finops(
    project_id: str,
    request: FinOpsIntegrationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Integrate migration project with FinOps platform.
    
    This endpoint configures cost tracking, budget alerts, and optimization
    features based on the organizational structure and resource categorization
    from the migration.
    
    Requirements: 7.1, 7.2
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
        
        # Initialize integration engine
        engine = PostMigrationIntegrationEngine(db)
        
        # Perform FinOps integration
        result = engine.integrate_with_finops(
            project_id=project_id,
            enable_cost_tracking=request.enable_cost_tracking,
            enable_budget_alerts=request.enable_budget_alerts,
            enable_waste_detection=request.enable_waste_detection,
            enable_optimization=request.enable_optimization,
            cost_allocation_method=request.cost_allocation_method
        )
        
        db.commit()
        
        logger.info(
            "FinOps integration completed",
            project_id=project_id,
            integration_id=result['integration_id'],
            features_enabled=len(result['optimization_features_enabled'])
        )
        
        return FinOpsIntegrationResponse(
            integration_id=result['integration_id'],
            status=result['status'],
            cost_tracking_enabled=result['cost_tracking_enabled'],
            budget_alerts_configured=result['budget_alerts_configured'],
            optimization_features_enabled=result['optimization_features_enabled'],
            integration_timestamp=result['integration_timestamp']
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
        logger.error("Failed to integrate with FinOps", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to integrate with FinOps: {str(e)}"
        )


@router.post("/{project_id}/integration/baselines", response_model=BaselineMetricsResponse)
async def capture_baseline_metrics(
    project_id: str,
    request: BaselineCaptureRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Capture baseline cost and performance metrics.
    
    This endpoint captures initial metrics after migration to establish
    baselines for future optimization and anomaly detection.
    
    Requirements: 7.3
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
        
        # Initialize integration engine
        engine = PostMigrationIntegrationEngine(db)
        
        # Capture baseline metrics
        baseline = engine.capture_baseline_metrics(
            project_id=project_id,
            capture_cost_data=request.capture_cost_data,
            capture_performance_data=request.capture_performance_data,
            capture_utilization_data=request.capture_utilization_data,
            baseline_period_days=request.baseline_period_days
        )
        
        db.commit()
        
        logger.info(
            "Baseline metrics captured",
            project_id=project_id,
            baseline_id=str(baseline.id),
            total_cost=float(baseline.total_monthly_cost),
            resource_count=baseline.resource_count
        )
        
        return BaselineMetricsResponse(
            baseline_id=str(baseline.id),
            capture_date=baseline.capture_date.isoformat(),
            total_monthly_cost=float(baseline.total_monthly_cost),
            cost_by_service=baseline.cost_by_service or {},
            cost_by_team=baseline.cost_by_team or {},
            cost_by_project=baseline.cost_by_project or {},
            cost_by_environment=baseline.cost_by_environment or {},
            resource_count=baseline.resource_count,
            resource_count_by_type=baseline.resource_count_by_type or {}
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
        logger.error("Failed to capture baseline metrics", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to capture baseline metrics: {str(e)}"
        )


@router.get("/{project_id}/integration/baselines")
async def get_baseline_metrics(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Retrieve baseline metrics for a migration project.
    
    Returns the most recent baseline metrics captured for the project.
    
    Requirements: 7.3
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
        
        # Get most recent baseline
        baseline = db.query(BaselineMetrics).filter(
            BaselineMetrics.migration_project_id == project.id
        ).order_by(BaselineMetrics.capture_date.desc()).first()
        
        if not baseline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No baseline metrics found for project {project_id}"
            )
        
        return BaselineMetricsResponse(
            baseline_id=str(baseline.id),
            capture_date=baseline.capture_date.isoformat(),
            total_monthly_cost=float(baseline.total_monthly_cost),
            cost_by_service=baseline.cost_by_service or {},
            cost_by_team=baseline.cost_by_team or {},
            cost_by_project=baseline.cost_by_project or {},
            cost_by_environment=baseline.cost_by_environment or {},
            resource_count=baseline.resource_count,
            resource_count_by_type=baseline.resource_count_by_type or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get baseline metrics", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get baseline metrics: {str(e)}"
        )


@router.get("/{project_id}/reports/final", response_model=MigrationReportResponse)
async def get_migration_report(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Generate and retrieve comprehensive migration report.
    
    This endpoint generates a complete migration report including costs,
    timeline analysis, success metrics, lessons learned, and optimization
    opportunities.
    
    Requirements: 7.5
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
        
        # Check if report already exists
        report = db.query(MigrationReport).filter(
            MigrationReport.migration_project_id == project.id
        ).first()
        
        if not report:
            # Generate new report
            engine = PostMigrationIntegrationEngine(db)
            report = engine.generate_migration_report(project_id=project_id)
            db.commit()
            
            logger.info(
                "Migration report generated",
                project_id=project_id,
                report_id=str(report.id),
                success_rate=report.success_rate
            )
        
        # Calculate cost variance
        cost_variance = 0.0
        if report.budgeted_cost and report.budgeted_cost > 0:
            cost_variance = ((float(report.total_cost) - float(report.budgeted_cost)) / float(report.budgeted_cost)) * 100
        
        return MigrationReportResponse(
            report_id=str(report.id),
            report_date=report.report_date.isoformat(),
            start_date=report.start_date.isoformat(),
            completion_date=report.completion_date.isoformat(),
            actual_duration_days=report.actual_duration_days,
            planned_duration_days=report.planned_duration_days,
            total_cost=float(report.total_cost),
            budgeted_cost=float(report.budgeted_cost),
            cost_variance_percentage=cost_variance,
            resources_migrated=report.resources_migrated,
            success_rate=report.success_rate,
            lessons_learned=report.lessons_learned or [],
            optimization_opportunities=report.optimization_opportunities or [],
            cost_analysis=report.cost_analysis or {},
            timeline_analysis=report.timeline_analysis or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to get migration report", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get migration report: {str(e)}"
        )


@router.post("/{project_id}/reports/final/regenerate", response_model=MigrationReportResponse)
async def regenerate_migration_report(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Regenerate migration report with latest data.
    
    This endpoint forces regeneration of the migration report, useful when
    additional data has been collected or corrections have been made.
    
    Requirements: 7.5
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
        
        # Delete existing report
        db.query(MigrationReport).filter(
            MigrationReport.migration_project_id == project.id
        ).delete()
        
        # Generate new report
        engine = PostMigrationIntegrationEngine(db)
        report = engine.generate_migration_report(project_id=project_id)
        
        db.commit()
        
        logger.info(
            "Migration report regenerated",
            project_id=project_id,
            report_id=str(report.id)
        )
        
        # Calculate cost variance
        cost_variance = 0.0
        if report.budgeted_cost and report.budgeted_cost > 0:
            cost_variance = ((float(report.total_cost) - float(report.budgeted_cost)) / float(report.budgeted_cost)) * 100
        
        return MigrationReportResponse(
            report_id=str(report.id),
            report_date=report.report_date.isoformat(),
            start_date=report.start_date.isoformat(),
            completion_date=report.completion_date.isoformat(),
            actual_duration_days=report.actual_duration_days,
            planned_duration_days=report.planned_duration_days,
            total_cost=float(report.total_cost),
            budgeted_cost=float(report.budgeted_cost),
            cost_variance_percentage=cost_variance,
            resources_migrated=report.resources_migrated,
            success_rate=report.success_rate,
            lessons_learned=report.lessons_learned or [],
            optimization_opportunities=report.optimization_opportunities or [],
            cost_analysis=report.cost_analysis or {},
            timeline_analysis=report.timeline_analysis or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to regenerate migration report", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate migration report: {str(e)}"
        )
