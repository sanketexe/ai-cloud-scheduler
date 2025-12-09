"""
API Endpoints for Workload and Requirements Analysis

This module provides REST API endpoints for workload profiling, performance requirements,
compliance assessment, budget analysis, and technical requirements mapping.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
import uuid

from ..database import get_db_session
from ..auth import get_current_user
from ..models import User
from .requirements_analysis_engine import WorkloadAnalysisEngine

router = APIRouter(prefix="/api/migrations", tags=["requirements-analysis"])


# Pydantic models for request/response

class WorkloadProfileRequest(BaseModel):
    """Request model for creating a workload profile"""
    workload_name: str = Field(..., min_length=1, max_length=255)
    application_type: str = Field(..., min_length=1, max_length=100)
    total_compute_cores: int | None = Field(None, ge=0)
    total_memory_gb: int | None = Field(None, ge=0)
    total_storage_tb: float | None = Field(None, ge=0)
    database_types: List[str] | None = None
    data_volume_tb: float | None = Field(None, ge=0)
    peak_transaction_rate: int | None = Field(None, ge=0)
    workload_patterns: Dict[str, Any] | None = None
    dependencies: List[str] | None = None


class PerformanceRequirementsRequest(BaseModel):
    """Request model for creating performance requirements"""
    availability_target: float = Field(..., ge=0, le=100)
    latency_requirements: Dict[str, Any] | None = None
    disaster_recovery_rto: int | None = Field(None, ge=0)
    disaster_recovery_rpo: int | None = Field(None, ge=0)
    geographic_distribution: List[str] | None = None
    peak_load_multiplier: float | None = Field(None, ge=1.0)
    additional_requirements: Dict[str, Any] | None = None


class ComplianceRequirementsRequest(BaseModel):
    """Request model for creating compliance requirements"""
    regulatory_frameworks: List[str] | None = None
    data_residency_requirements: List[str] | None = None
    industry_certifications: List[str] | None = None
    security_standards: List[str] | None = None
    audit_requirements: Dict[str, Any] | None = None
    additional_compliance: Dict[str, Any] | None = None


class BudgetConstraintsRequest(BaseModel):
    """Request model for creating budget constraints"""
    migration_budget: float = Field(..., gt=0)
    current_monthly_cost: float | None = Field(None, ge=0)
    target_monthly_cost: float | None = Field(None, ge=0)
    cost_optimization_priority: str = Field(default="medium")
    acceptable_cost_variance: float | None = Field(None, ge=0, le=100)
    currency: str = Field(default="USD", max_length=3)
    additional_constraints: Dict[str, Any] | None = None
    
    @validator('cost_optimization_priority')
    def validate_priority(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError('Priority must be low, medium, or high')
        return v


class TechnicalRequirementsRequest(BaseModel):
    """Request model for creating technical requirements"""
    required_services: List[str] | None = None
    ml_ai_requirements: Dict[str, Any] | None = None
    analytics_requirements: Dict[str, Any] | None = None
    container_orchestration: bool = False
    serverless_requirements: bool = False
    specialized_compute: List[str] | None = None
    integration_requirements: Dict[str, Any] | None = None
    additional_technical: Dict[str, Any] | None = None


# API Endpoints

@router.post("/{project_id}/workloads")
def create_workload_profile(
    project_id: str,
    workload_data: WorkloadProfileRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create a workload profile for a migration project.
    
    Implements Requirement: 2.1
    """
    try:
        engine = WorkloadAnalysisEngine(db)
        result = engine.analyze_workloads(
            project_id=project_id,
            workload_data=workload_data.dict(exclude_none=True)
        )
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{project_id}/workloads")
def get_workload_profiles(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get all workload profiles for a migration project.
    
    Implements Requirement: 2.1
    """
    try:
        from .models import MigrationProject, WorkloadProfile
        
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        
        workloads = db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == project.id
        ).all()
        
        return {
            'workloads': [
                {
                    'id': str(w.id),
                    'workload_name': w.workload_name,
                    'application_type': w.application_type,
                    'total_compute_cores': w.total_compute_cores,
                    'total_memory_gb': w.total_memory_gb,
                    'total_storage_tb': w.total_storage_tb,
                    'database_types': w.database_types,
                    'data_volume_tb': w.data_volume_tb,
                    'peak_transaction_rate': w.peak_transaction_rate,
                    'workload_patterns': w.workload_patterns,
                    'dependencies': w.dependencies
                }
                for w in workloads
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{project_id}/performance-requirements")
def create_performance_requirements(
    project_id: str,
    perf_data: PerformanceRequirementsRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create performance requirements for a migration project.
    
    Implements Requirement: 2.2
    """
    try:
        engine = WorkloadAnalysisEngine(db)
        result = engine.assess_performance_requirements(
            project_id=project_id,
            perf_requirements=perf_data.dict(exclude_none=True)
        )
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{project_id}/performance-requirements")
def get_performance_requirements(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get performance requirements for a migration project.
    
    Implements Requirement: 2.2
    """
    try:
        from .models import MigrationProject, PerformanceRequirements
        
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        
        perf_req = db.query(PerformanceRequirements).filter(
            PerformanceRequirements.migration_project_id == project.id
        ).first()
        
        if not perf_req:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Performance requirements not found")
        
        return {
            'id': str(perf_req.id),
            'availability_target': float(perf_req.availability_target),
            'latency_requirements': perf_req.latency_requirements,
            'disaster_recovery_rto': perf_req.disaster_recovery_rto,
            'disaster_recovery_rpo': perf_req.disaster_recovery_rpo,
            'geographic_distribution': perf_req.geographic_distribution,
            'peak_load_multiplier': perf_req.peak_load_multiplier,
            'additional_requirements': perf_req.additional_requirements
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{project_id}/compliance-requirements")
def create_compliance_requirements(
    project_id: str,
    compliance_data: ComplianceRequirementsRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create compliance requirements for a migration project.
    
    Implements Requirement: 2.3
    """
    try:
        engine = WorkloadAnalysisEngine(db)
        result = engine.evaluate_compliance_needs(
            project_id=project_id,
            compliance_data=compliance_data.dict(exclude_none=True)
        )
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{project_id}/compliance-requirements")
def get_compliance_requirements(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get compliance requirements for a migration project.
    
    Implements Requirement: 2.3
    """
    try:
        from .models import MigrationProject, ComplianceRequirements
        
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        
        compliance_req = db.query(ComplianceRequirements).filter(
            ComplianceRequirements.migration_project_id == project.id
        ).first()
        
        if not compliance_req:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Compliance requirements not found")
        
        return {
            'id': str(compliance_req.id),
            'regulatory_frameworks': compliance_req.regulatory_frameworks,
            'data_residency_requirements': compliance_req.data_residency_requirements,
            'industry_certifications': compliance_req.industry_certifications,
            'security_standards': compliance_req.security_standards,
            'audit_requirements': compliance_req.audit_requirements,
            'additional_compliance': compliance_req.additional_compliance
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{project_id}/budget-constraints")
def create_budget_constraints(
    project_id: str,
    budget_data: BudgetConstraintsRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create budget constraints for a migration project.
    
    Implements Requirement: 2.4
    """
    try:
        engine = WorkloadAnalysisEngine(db)
        result = engine.analyze_budget_constraints(
            project_id=project_id,
            budget_data=budget_data.dict(exclude_none=True)
        )
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{project_id}/budget-constraints")
def get_budget_constraints(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get budget constraints for a migration project.
    
    Implements Requirement: 2.4
    """
    try:
        from .models import MigrationProject, BudgetConstraints
        
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        
        budget = db.query(BudgetConstraints).filter(
            BudgetConstraints.migration_project_id == project.id
        ).first()
        
        if not budget:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Budget constraints not found")
        
        return {
            'id': str(budget.id),
            'migration_budget': float(budget.migration_budget),
            'current_monthly_cost': float(budget.current_monthly_cost) if budget.current_monthly_cost else None,
            'target_monthly_cost': float(budget.target_monthly_cost) if budget.target_monthly_cost else None,
            'cost_optimization_priority': budget.cost_optimization_priority,
            'acceptable_cost_variance': budget.acceptable_cost_variance,
            'currency': budget.currency,
            'additional_constraints': budget.additional_constraints
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{project_id}/technical-requirements")
def create_technical_requirements(
    project_id: str,
    tech_data: TechnicalRequirementsRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create technical requirements for a migration project.
    
    Implements Requirement: 2.5
    """
    try:
        engine = WorkloadAnalysisEngine(db)
        result = engine.map_technical_requirements(
            project_id=project_id,
            tech_requirements=tech_data.dict(exclude_none=True)
        )
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{project_id}/technical-requirements")
def get_technical_requirements(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get technical requirements for a migration project.
    
    Implements Requirement: 2.5
    """
    try:
        from .models import MigrationProject, TechnicalRequirements
        
        project = db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        
        tech_req = db.query(TechnicalRequirements).filter(
            TechnicalRequirements.migration_project_id == project.id
        ).first()
        
        if not tech_req:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Technical requirements not found")
        
        return {
            'id': str(tech_req.id),
            'required_services': tech_req.required_services,
            'ml_ai_requirements': tech_req.ml_ai_requirements,
            'analytics_requirements': tech_req.analytics_requirements,
            'container_orchestration': tech_req.container_orchestration,
            'serverless_requirements': tech_req.serverless_requirements,
            'specialized_compute': tech_req.specialized_compute,
            'integration_requirements': tech_req.integration_requirements,
            'additional_technical': tech_req.additional_technical
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{project_id}/requirements/validation")
def validate_requirements_completeness(
    project_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Validate completeness and consistency of all requirements for a migration project.
    
    Implements Requirement: 2.6
    """
    try:
        engine = WorkloadAnalysisEngine(db)
        result = engine.validate_requirements_completeness(project_id=project_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
