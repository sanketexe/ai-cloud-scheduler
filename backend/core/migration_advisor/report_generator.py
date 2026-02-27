"""
Comprehensive report generation service for migration recommendations.

This module provides functionality to generate detailed migration reports
including executive summaries, technical analysis, and implementation roadmaps.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from decimal import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import and_

from .models import (
    MigrationProject, 
    RecommendationReport as DBRecommendationReport,
    ProviderEvaluation,
    OrganizationProfile,
    WorkloadProfile,
    PerformanceRequirements,
    ComplianceRequirements,
    BudgetConstraints,
    TechnicalRequirements
)
from .recommendation_engine import RecommendationEngine
from .provider_catalog import CloudProviderName


@dataclass
class ExecutiveSummary:
    """Executive summary section of the report"""
    organization_name: str
    assessment_date: datetime
    primary_recommendation: str
    estimated_monthly_cost: Optional[float]
    estimated_savings: Optional[float]
    migration_duration_weeks: int
    confidence_score: float
    key_benefits: List[str]
    critical_considerations: List[str]


@dataclass
class TechnicalAnalysis:
    """Technical analysis section of the report"""
    workload_summary: Dict[str, Any]
    performance_requirements: Dict[str, Any]
    compliance_requirements: Dict[str, Any]
    technical_constraints: Dict[str, Any]
    provider_evaluations: List[Dict[str, Any]]
    comparison_matrix: Dict[str, Any]
    risk_assessment: Dict[str, Any]


@dataclass
class ImplementationRoadmap:
    """Implementation roadmap section of the report"""
    migration_phases: List[Dict[str, Any]]
    timeline_overview: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    success_criteria: List[str]
    potential_challenges: List[str]
    mitigation_strategies: List[str]


@dataclass
class AssessmentInputs:
    """Assessment inputs section for transparency"""
    organization_profile: Dict[str, Any]
    workload_profile: Dict[str, Any]
    requirements_summary: Dict[str, Any]
    scoring_methodology: Dict[str, Any]
    assumptions: List[str]


@dataclass
class ComprehensiveReport:
    """Complete migration recommendation report"""
    report_id: str
    project_id: str
    generated_at: datetime
    executive_summary: ExecutiveSummary
    technical_analysis: TechnicalAnalysis
    implementation_roadmap: ImplementationRoadmap
    assessment_inputs: AssessmentInputs
    appendices: Dict[str, Any]


class ReportGenerator:
    """Service for generating comprehensive migration reports"""
    
    def __init__(self, db: Session):
        self.db = db
        self.recommendation_engine = RecommendationEngine()
    
    def generate_comprehensive_report(self, project_id: str) -> ComprehensiveReport:
        """
        Generate a comprehensive migration recommendation report.
        
        Args:
            project_id: The migration project ID
            
        Returns:
            ComprehensiveReport: Complete report with all sections
            
        Raises:
            ValueError: If project not found or incomplete
        """
        # Load project and validate completeness
        project = self._load_and_validate_project(project_id)
        
        # Get recommendation report
        recommendation_report = self._get_recommendation_report(project_id)
        
        # Get provider evaluations
        evaluations = self._get_provider_evaluations(project_id)
        
        # Generate report sections
        executive_summary = self._generate_executive_summary(project, recommendation_report, evaluations)
        technical_analysis = self._generate_technical_analysis(project, evaluations, recommendation_report)
        implementation_roadmap = self._generate_implementation_roadmap(project, recommendation_report)
        assessment_inputs = self._generate_assessment_inputs(project, recommendation_report)
        appendices = self._generate_appendices(project, evaluations)
        
        return ComprehensiveReport(
            report_id=str(uuid.uuid4()),
            project_id=project_id,
            generated_at=datetime.utcnow(),
            executive_summary=executive_summary,
            technical_analysis=technical_analysis,
            implementation_roadmap=implementation_roadmap,
            assessment_inputs=assessment_inputs,
            appendices=appendices
        )
    
    def _load_and_validate_project(self, project_id: str) -> MigrationProject:
        """Load project and validate it has all required data"""
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Check for required components
        required_components = [
            project.organization_profile,
            project.workload_profile,
            project.performance_requirements,
            project.compliance_requirements,
            project.budget_constraints,
            project.technical_requirements
        ]
        
        if not all(required_components):
            raise ValueError(f"Project {project_id} is incomplete - missing required assessment data")
        
        return project
    
    def _get_recommendation_report(self, project_id: str) -> DBRecommendationReport:
        """Get the recommendation report for the project"""
        report = self.db.query(DBRecommendationReport).filter(
            DBRecommendationReport.migration_project_id == project_id
        ).first()
        
        if not report:
            raise ValueError(f"No recommendation report found for project {project_id}")
        
        return report
    
    def _get_provider_evaluations(self, project_id: str) -> List[ProviderEvaluation]:
        """Get all provider evaluations for the project"""
        evaluations = self.db.query(ProviderEvaluation).filter(
            ProviderEvaluation.migration_project_id == project_id
        ).order_by(ProviderEvaluation.overall_score.desc()).all()
        
        return evaluations
    
    def _generate_executive_summary(
        self, 
        project: MigrationProject, 
        recommendation_report: DBRecommendationReport,
        evaluations: List[ProviderEvaluation]
    ) -> ExecutiveSummary:
        """Generate executive summary section"""
        
        # Find primary recommendation evaluation
        primary_eval = next(
            (e for e in evaluations if e.provider_name.value == recommendation_report.primary_recommendation),
            evaluations[0] if evaluations else None
        )
        
        # Calculate estimated savings
        estimated_savings = None
        if len(evaluations) >= 2:
            highest_cost = max(e.estimated_monthly_cost or 0 for e in evaluations)
            primary_cost = primary_eval.estimated_monthly_cost or 0
            if highest_cost > primary_cost:
                estimated_savings = float(highest_cost - primary_cost)
        
        # Generate key benefits
        key_benefits = []
        if primary_eval:
            key_benefits.extend(primary_eval.strengths or [])
            if estimated_savings and estimated_savings > 0:
                key_benefits.append(f"Potential monthly savings of ${estimated_savings:,.2f}")
        
        # Generate critical considerations
        critical_considerations = []
        if primary_eval:
            critical_considerations.extend(primary_eval.weaknesses or [])
        
        return ExecutiveSummary(
            organization_name=project.organization_profile.company_name,
            assessment_date=project.created_at,
            primary_recommendation=recommendation_report.primary_recommendation,
            estimated_monthly_cost=float(primary_eval.estimated_monthly_cost) if primary_eval and primary_eval.estimated_monthly_cost else None,
            estimated_savings=estimated_savings,
            migration_duration_weeks=primary_eval.migration_duration_weeks if primary_eval else 12,
            confidence_score=recommendation_report.confidence_score,
            key_benefits=key_benefits[:5],  # Top 5 benefits
            critical_considerations=critical_considerations[:3]  # Top 3 considerations
        )
    
    def _generate_technical_analysis(
        self,
        project: MigrationProject,
        evaluations: List[ProviderEvaluation],
        recommendation_report: DBRecommendationReport
    ) -> TechnicalAnalysis:
        """Generate technical analysis section"""
        
        # Workload summary
        workload_summary = {
            "total_compute_cores": project.workload_profile.total_compute_cores,
            "total_memory_gb": project.workload_profile.total_memory_gb,
            "total_storage_tb": float(project.workload_profile.total_storage_tb),
            "database_types": project.workload_profile.database_types,
            "peak_transaction_rate": project.workload_profile.peak_transaction_rate,
            "application_count": len(project.workload_profile.applications or [])
        }
        
        # Performance requirements
        performance_requirements = {
            "availability_target": float(project.performance_requirements.availability_target),
            "max_latency_ms": project.performance_requirements.max_latency_ms,
            "throughput_requirements": project.performance_requirements.throughput_requirements,
            "scalability_requirements": project.performance_requirements.scalability_requirements
        }
        
        # Compliance requirements
        compliance_requirements = {
            "regulatory_frameworks": project.compliance_requirements.regulatory_frameworks,
            "data_residency_requirements": project.compliance_requirements.data_residency_requirements,
            "industry_certifications": project.compliance_requirements.industry_certifications,
            "security_standards": project.compliance_requirements.security_standards
        }
        
        # Technical constraints
        technical_constraints = {
            "preferred_technologies": project.technical_requirements.preferred_technologies,
            "integration_requirements": project.technical_requirements.integration_requirements,
            "security_requirements": project.technical_requirements.security_requirements,
            "backup_requirements": project.technical_requirements.backup_requirements
        }
        
        # Provider evaluations
        provider_evaluations = []
        for eval in evaluations:
            provider_evaluations.append({
                "provider": eval.provider_name.value,
                "overall_score": float(eval.overall_score),
                "cost_score": float(eval.cost_score),
                "performance_score": float(eval.performance_score),
                "compliance_score": float(eval.compliance_score),
                "migration_complexity_score": float(eval.migration_complexity_score),
                "estimated_monthly_cost": float(eval.estimated_monthly_cost) if eval.estimated_monthly_cost else None,
                "migration_duration_weeks": eval.migration_duration_weeks,
                "strengths": eval.strengths,
                "weaknesses": eval.weaknesses
            })
        
        # Comparison matrix
        comparison_matrix = {
            "key_differences": recommendation_report.key_differentiators,
            "cost_comparison": recommendation_report.cost_comparison,
            "scoring_weights": recommendation_report.scoring_weights
        }
        
        # Risk assessment
        risk_assessment = recommendation_report.risk_assessment or {}
        
        return TechnicalAnalysis(
            workload_summary=workload_summary,
            performance_requirements=performance_requirements,
            compliance_requirements=compliance_requirements,
            technical_constraints=technical_constraints,
            provider_evaluations=provider_evaluations,
            comparison_matrix=comparison_matrix,
            risk_assessment=risk_assessment
        )
    
    def _generate_implementation_roadmap(
        self,
        project: MigrationProject,
        recommendation_report: DBRecommendationReport
    ) -> ImplementationRoadmap:
        """Generate implementation roadmap section"""
        
        # Migration phases (simplified for now)
        migration_phases = [
            {
                "phase": "Assessment & Planning",
                "duration_weeks": 2,
                "description": "Detailed assessment and migration planning",
                "key_activities": [
                    "Infrastructure assessment",
                    "Application dependency mapping",
                    "Migration strategy finalization",
                    "Team training"
                ]
            },
            {
                "phase": "Environment Setup",
                "duration_weeks": 3,
                "description": "Cloud environment preparation",
                "key_activities": [
                    "Cloud account setup",
                    "Network configuration",
                    "Security implementation",
                    "Monitoring setup"
                ]
            },
            {
                "phase": "Pilot Migration",
                "duration_weeks": 4,
                "description": "Migrate non-critical workloads first",
                "key_activities": [
                    "Select pilot applications",
                    "Execute pilot migration",
                    "Performance testing",
                    "Process refinement"
                ]
            },
            {
                "phase": "Production Migration",
                "duration_weeks": 8,
                "description": "Migrate production workloads",
                "key_activities": [
                    "Staged migration execution",
                    "Data synchronization",
                    "Cutover coordination",
                    "Validation testing"
                ]
            },
            {
                "phase": "Optimization & Closure",
                "duration_weeks": 3,
                "description": "Post-migration optimization",
                "key_activities": [
                    "Performance optimization",
                    "Cost optimization",
                    "Documentation",
                    "Knowledge transfer"
                ]
            }
        ]
        
        # Timeline overview
        total_weeks = sum(phase["duration_weeks"] for phase in migration_phases)
        timeline_overview = {
            "total_duration_weeks": total_weeks,
            "estimated_start_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "estimated_completion_date": (datetime.utcnow() + timedelta(weeks=total_weeks)).strftime("%Y-%m-%d"),
            "critical_path": ["Environment Setup", "Production Migration"]
        }
        
        # Resource requirements
        resource_requirements = {
            "project_manager": "1 FTE for entire duration",
            "cloud_architect": "1 FTE for first 8 weeks",
            "migration_engineers": "2-3 FTE for weeks 3-17",
            "application_teams": "Variable based on application complexity",
            "estimated_budget": f"${float(project.budget_constraints.migration_budget):,.2f}" if project.budget_constraints.migration_budget else "TBD"
        }
        
        # Success criteria
        success_criteria = [
            "All applications successfully migrated with <2% downtime",
            "Performance meets or exceeds current baseline",
            "Security and compliance requirements satisfied",
            "Migration completed within budget",
            "Team trained on new cloud environment"
        ]
        
        # Potential challenges
        potential_challenges = [
            "Application dependencies and integration complexity",
            "Data migration for large datasets",
            "Network connectivity and performance",
            "Team learning curve on new platform",
            "Compliance validation in cloud environment"
        ]
        
        # Mitigation strategies
        mitigation_strategies = [
            "Comprehensive dependency mapping and testing",
            "Phased migration approach with rollback plans",
            "Early network setup and performance testing",
            "Structured training program and documentation",
            "Regular compliance audits throughout migration"
        ]
        
        return ImplementationRoadmap(
            migration_phases=migration_phases,
            timeline_overview=timeline_overview,
            resource_requirements=resource_requirements,
            success_criteria=success_criteria,
            potential_challenges=potential_challenges,
            mitigation_strategies=mitigation_strategies
        )
    
    def _generate_assessment_inputs(
        self,
        project: MigrationProject,
        recommendation_report: DBRecommendationReport
    ) -> AssessmentInputs:
        """Generate assessment inputs section for transparency"""
        
        # Organization profile
        organization_profile = {
            "company_name": project.organization_profile.company_name,
            "company_size": project.organization_profile.company_size.value,
            "industry": project.organization_profile.industry,
            "current_infrastructure": project.organization_profile.current_infrastructure.value,
            "it_team_size": project.organization_profile.it_team_size,
            "cloud_experience_level": project.organization_profile.cloud_experience_level.value,
            "geographic_presence": project.organization_profile.geographic_presence
        }
        
        # Workload profile
        workload_profile = {
            "total_compute_cores": project.workload_profile.total_compute_cores,
            "total_memory_gb": project.workload_profile.total_memory_gb,
            "total_storage_tb": float(project.workload_profile.total_storage_tb),
            "database_types": project.workload_profile.database_types,
            "peak_transaction_rate": project.workload_profile.peak_transaction_rate,
            "applications": project.workload_profile.applications
        }
        
        # Requirements summary
        requirements_summary = {
            "performance": {
                "availability_target": float(project.performance_requirements.availability_target),
                "max_latency_ms": project.performance_requirements.max_latency_ms
            },
            "compliance": {
                "regulatory_frameworks": project.compliance_requirements.regulatory_frameworks,
                "data_residency": project.compliance_requirements.data_residency_requirements
            },
            "budget": {
                "migration_budget": float(project.budget_constraints.migration_budget),
                "target_monthly_cost": float(project.budget_constraints.target_monthly_cost) if project.budget_constraints.target_monthly_cost else None
            },
            "technical": {
                "preferred_technologies": project.technical_requirements.preferred_technologies,
                "security_requirements": project.technical_requirements.security_requirements
            }
        }
        
        # Scoring methodology
        scoring_methodology = {
            "weights_used": recommendation_report.scoring_weights,
            "evaluation_criteria": [
                "Cost effectiveness (30%)",
                "Performance capability (25%)",
                "Compliance support (20%)",
                "Migration complexity (15%)",
                "Service availability (10%)"
            ],
            "scoring_scale": "0-100 points per criterion"
        }
        
        # Assumptions
        assumptions = [
            "Current infrastructure costs estimated based on provided data",
            "Migration timeline assumes dedicated team availability",
            "Cost estimates based on current cloud pricing (subject to change)",
            "Performance requirements validated against current baseline",
            "Compliance requirements verified against provider certifications"
        ]
        
        return AssessmentInputs(
            organization_profile=organization_profile,
            workload_profile=workload_profile,
            requirements_summary=requirements_summary,
            scoring_methodology=scoring_methodology,
            assumptions=assumptions
        )
    
    def _generate_appendices(
        self,
        project: MigrationProject,
        evaluations: List[ProviderEvaluation]
    ) -> Dict[str, Any]:
        """Generate appendices with additional details"""
        
        return {
            "detailed_cost_breakdown": {
                provider.provider_name.value: {
                    "compute_cost": float(provider.estimated_monthly_cost * Decimal('0.4')) if provider.estimated_monthly_cost else 0,
                    "storage_cost": float(provider.estimated_monthly_cost * Decimal('0.25')) if provider.estimated_monthly_cost else 0,
                    "network_cost": float(provider.estimated_monthly_cost * Decimal('0.15')) if provider.estimated_monthly_cost else 0,
                    "database_cost": float(provider.estimated_monthly_cost * Decimal('0.15')) if provider.estimated_monthly_cost else 0,
                    "other_services": float(provider.estimated_monthly_cost * Decimal('0.05')) if provider.estimated_monthly_cost else 0
                }
                for provider in evaluations
            },
            "service_mapping": {
                "compute": ["EC2/Compute Engine/Virtual Machines"],
                "storage": ["S3/Cloud Storage/Blob Storage"],
                "database": ["RDS/Cloud SQL/Azure Database"],
                "networking": ["VPC/VPC/Virtual Network"]
            },
            "compliance_matrix": {
                framework: {
                    provider.provider_name.value: provider.compliance_score >= 80
                    for provider in evaluations
                }
                for framework in project.compliance_requirements.regulatory_frameworks or []
            }
        }


class ShareableLinkManager:
    """Manages shareable links for migration reports"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_shareable_link(self, project_id: str, expires_in_days: int = 30) -> str:
        """
        Create a shareable link for a migration report.
        
        Args:
            project_id: The migration project ID
            expires_in_days: Number of days until link expires
            
        Returns:
            str: Shareable link token
        """
        # Generate unique token
        link_token = str(uuid.uuid4())
        
        # Store link information (in a real implementation, you'd store this in a database)
        # For now, we'll use a simple approach
        link_data = {
            "project_id": project_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat(),
            "access_count": 0
        }
        
        # In a real implementation, store this in a database table
        # For now, we'll return the token
        return link_token
    
    def validate_shareable_link(self, link_token: str) -> Optional[str]:
        """
        Validate a shareable link and return the project ID if valid.
        
        Args:
            link_token: The shareable link token
            
        Returns:
            Optional[str]: Project ID if link is valid, None otherwise
        """
        # In a real implementation, look up the token in the database
        # and check if it's still valid (not expired)
        # For now, we'll return None to indicate this needs implementation
        return None