"""
API endpoints for migration report generation and export.

This module provides REST API endpoints for generating comprehensive reports,
exporting to PDF, and creating shareable links.
"""

import io
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database import get_db_session
from ..auth import get_current_user
from ..models import User
from .models import MigrationProject
from .report_generator import ReportGenerator, ShareableLinkManager, ComprehensiveReport
from .pdf_export_service import export_report_to_pdf, REPORTLAB_AVAILABLE


router = APIRouter(prefix="/migration-advisor", tags=["Migration Reports"])


# Response Models

class ReportSummaryResponse(BaseModel):
    """Summary response for report generation"""
    report_id: str
    project_id: str
    generated_at: datetime
    primary_recommendation: str
    confidence_score: float
    estimated_monthly_cost: Optional[float]
    estimated_savings: Optional[float]


class ShareableLinkResponse(BaseModel):
    """Response for shareable link creation"""
    link_token: str
    expires_at: datetime
    share_url: str


class ReportStatusResponse(BaseModel):
    """Response for report generation status"""
    status: str  # "generating", "completed", "failed"
    message: str
    report_id: Optional[str] = None


# API Endpoints

@router.post("/{project_id}/reports/generate", response_model=ReportSummaryResponse)
async def generate_comprehensive_report(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Generate a comprehensive migration recommendation report.
    
    This endpoint creates a detailed report including:
    - Executive summary with key recommendations
    - Technical analysis of workloads and requirements
    - Implementation roadmap with phases and timelines
    - Assessment inputs for transparency
    - Detailed appendices with cost breakdowns
    
    The report can be used for stakeholder presentations and migration planning.
    """
    try:
        # Verify project exists and user has access
        project = db.query(MigrationProject).filter(
            MigrationProject.id == project_id,
            MigrationProject.user_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Migration project {project_id} not found"
            )
        
        # Generate comprehensive report
        report_generator = ReportGenerator(db)
        report = report_generator.generate_comprehensive_report(project_id)
        
        return ReportSummaryResponse(
            report_id=report.report_id,
            project_id=report.project_id,
            generated_at=report.generated_at,
            primary_recommendation=report.executive_summary.primary_recommendation,
            confidence_score=report.executive_summary.confidence_score,
            estimated_monthly_cost=report.executive_summary.estimated_monthly_cost,
            estimated_savings=report.executive_summary.estimated_savings
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/{project_id}/reports/latest")
async def get_latest_report(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get the latest comprehensive report for a project.
    
    Returns the full report data including all sections:
    executive summary, technical analysis, implementation roadmap,
    and assessment inputs.
    """
    try:
        # Verify project exists and user has access
        project = db.query(MigrationProject).filter(
            MigrationProject.id == project_id,
            MigrationProject.user_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Migration project {project_id} not found"
            )
        
        # Generate report (in a real implementation, you might cache this)
        report_generator = ReportGenerator(db)
        report = report_generator.generate_comprehensive_report(project_id)
        
        # Convert to dict for JSON response
        return {
            "report_id": report.report_id,
            "project_id": report.project_id,
            "generated_at": report.generated_at.isoformat(),
            "executive_summary": {
                "organization_name": report.executive_summary.organization_name,
                "assessment_date": report.executive_summary.assessment_date.isoformat(),
                "primary_recommendation": report.executive_summary.primary_recommendation,
                "estimated_monthly_cost": report.executive_summary.estimated_monthly_cost,
                "estimated_savings": report.executive_summary.estimated_savings,
                "migration_duration_weeks": report.executive_summary.migration_duration_weeks,
                "confidence_score": report.executive_summary.confidence_score,
                "key_benefits": report.executive_summary.key_benefits,
                "critical_considerations": report.executive_summary.critical_considerations
            },
            "technical_analysis": {
                "workload_summary": report.technical_analysis.workload_summary,
                "performance_requirements": report.technical_analysis.performance_requirements,
                "compliance_requirements": report.technical_analysis.compliance_requirements,
                "technical_constraints": report.technical_analysis.technical_constraints,
                "provider_evaluations": report.technical_analysis.provider_evaluations,
                "comparison_matrix": report.technical_analysis.comparison_matrix,
                "risk_assessment": report.technical_analysis.risk_assessment
            },
            "implementation_roadmap": {
                "migration_phases": report.implementation_roadmap.migration_phases,
                "timeline_overview": report.implementation_roadmap.timeline_overview,
                "resource_requirements": report.implementation_roadmap.resource_requirements,
                "success_criteria": report.implementation_roadmap.success_criteria,
                "potential_challenges": report.implementation_roadmap.potential_challenges,
                "mitigation_strategies": report.implementation_roadmap.mitigation_strategies
            },
            "assessment_inputs": {
                "organization_profile": report.assessment_inputs.organization_profile,
                "workload_profile": report.assessment_inputs.workload_profile,
                "requirements_summary": report.assessment_inputs.requirements_summary,
                "scoring_methodology": report.assessment_inputs.scoring_methodology,
                "assumptions": report.assessment_inputs.assumptions
            },
            "appendices": report.appendices
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve report: {str(e)}"
        )


@router.get("/{project_id}/reports/export/pdf")
async def export_report_pdf(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Export the migration report as a PDF file.
    
    Returns a PDF document containing the complete migration recommendation
    report formatted for printing and sharing with stakeholders.
    
    Requires the reportlab library to be installed.
    """
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="PDF export is not available. ReportLab library is required."
        )
    
    try:
        # Verify project exists and user has access
        project = db.query(MigrationProject).filter(
            MigrationProject.id == project_id,
            MigrationProject.user_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Migration project {project_id} not found"
            )
        
        # Generate comprehensive report
        report_generator = ReportGenerator(db)
        report = report_generator.generate_comprehensive_report(project_id)
        
        # Export to PDF
        pdf_bytes = export_report_to_pdf(report)
        
        # Create filename
        org_name = report.executive_summary.organization_name.replace(" ", "_")
        filename = f"Migration_Report_{org_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        # Return PDF as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export PDF: {str(e)}"
        )


@router.post("/{project_id}/reports/share", response_model=ShareableLinkResponse)
async def create_shareable_link(
    project_id: str,
    expires_in_days: int = 30,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create a shareable link for the migration report.
    
    The link allows external stakeholders to view the report without
    requiring authentication. Links expire after the specified number of days.
    
    Args:
        expires_in_days: Number of days until the link expires (default: 30)
    """
    try:
        # Verify project exists and user has access
        project = db.query(MigrationProject).filter(
            MigrationProject.id == project_id,
            MigrationProject.user_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Migration project {project_id} not found"
            )
        
        # Create shareable link
        link_manager = ShareableLinkManager(db)
        link_token = link_manager.create_shareable_link(project_id, expires_in_days)
        
        # Calculate expiration date
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create share URL (in a real implementation, use your domain)
        share_url = f"https://your-domain.com/shared-reports/{link_token}"
        
        return ShareableLinkResponse(
            link_token=link_token,
            expires_at=expires_at,
            share_url=share_url
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create shareable link: {str(e)}"
        )


@router.get("/shared/{link_token}")
async def view_shared_report(
    link_token: str,
    db: Session = Depends(get_db_session)
):
    """
    View a shared migration report using a shareable link.
    
    This endpoint allows external users to view reports without authentication
    using a valid shareable link token.
    """
    try:
        # Validate shareable link
        link_manager = ShareableLinkManager(db)
        project_id = link_manager.validate_shareable_link(link_token)
        
        if not project_id:
            raise HTTPException(
                status_code=404,
                detail="Invalid or expired shareable link"
            )
        
        # Generate report (without user authentication)
        report_generator = ReportGenerator(db)
        report = report_generator.generate_comprehensive_report(project_id)
        
        # Return public view of report (might exclude sensitive data)
        return {
            "report_id": report.report_id,
            "generated_at": report.generated_at.isoformat(),
            "executive_summary": {
                "organization_name": report.executive_summary.organization_name,
                "primary_recommendation": report.executive_summary.primary_recommendation,
                "estimated_monthly_cost": report.executive_summary.estimated_monthly_cost,
                "estimated_savings": report.executive_summary.estimated_savings,
                "migration_duration_weeks": report.executive_summary.migration_duration_weeks,
                "confidence_score": report.executive_summary.confidence_score,
                "key_benefits": report.executive_summary.key_benefits
            },
            "technical_analysis": {
                "provider_evaluations": report.technical_analysis.provider_evaluations,
                "comparison_matrix": report.technical_analysis.comparison_matrix
            },
            "implementation_roadmap": {
                "timeline_overview": report.implementation_roadmap.timeline_overview,
                "migration_phases": report.implementation_roadmap.migration_phases
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve shared report: {str(e)}"
        )


@router.get("/{project_id}/reports/status", response_model=ReportStatusResponse)
async def get_report_status(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of report generation for a project.
    
    This endpoint can be used to check if a report is ready,
    still being generated, or if generation failed.
    """
    try:
        # Verify project exists and user has access
        project = db.query(MigrationProject).filter(
            MigrationProject.id == project_id,
            MigrationProject.user_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Migration project {project_id} not found"
            )
        
        # Check if project has recommendations (required for report)
        if not project.recommendation_report:
            return ReportStatusResponse(
                status="not_ready",
                message="Project assessment is incomplete. Complete all assessment steps to generate report."
            )
        
        # In a real implementation, you might track report generation status
        # For now, assume report can always be generated if recommendations exist
        return ReportStatusResponse(
            status="ready",
            message="Report can be generated",
            report_id=str(uuid.uuid4())  # Would be actual report ID in real implementation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check report status: {str(e)}"
        )