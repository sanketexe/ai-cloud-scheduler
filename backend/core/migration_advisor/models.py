"""
SQLAlchemy models for Cloud Migration Advisor

This module defines the database schema for migration projects, assessments,
recommendations, and resource organization.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional

from sqlalchemy import (
    Column, String, DateTime, Boolean, Text, Date, Numeric, 
    ForeignKey, Index, JSON, Enum, Integer, Float
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from ..database import Base


# Enums for Migration Advisor

class MigrationStatus(PyEnum):
    """Migration project status"""
    ASSESSMENT = "assessment"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    PLANNING = "planning"
    EXECUTION = "execution"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


class CompanySize(PyEnum):
    """Organization size categories"""
    SMALL = "small"  # < 50 employees
    MEDIUM = "medium"  # 50-500 employees
    LARGE = "large"  # 500-5000 employees
    ENTERPRISE = "enterprise"  # > 5000 employees


class InfrastructureType(PyEnum):
    """Current infrastructure type"""
    ON_PREMISES = "on_premises"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    MULTI_CLOUD = "multi_cloud"


class ExperienceLevel(PyEnum):
    """Cloud experience level"""
    NONE = "none"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PhaseStatus(PyEnum):
    """Migration phase status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class OwnershipStatus(PyEnum):
    """Resource ownership status"""
    ASSIGNED = "assigned"
    UNASSIGNED = "unassigned"
    PENDING = "pending"


class MigrationRiskLevel(PyEnum):
    """Migration risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Core Migration Models

class MigrationProject(Base):
    """Core migration project entity"""
    __tablename__ = "migration_projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(String(100), unique=True, nullable=False, index=True)
    organization_name = Column(String(255), nullable=False)
    status = Column(Enum(MigrationStatus), nullable=False, default=MigrationStatus.ASSESSMENT)
    current_phase = Column(String(100))
    estimated_completion = Column(DateTime(timezone=True))
    actual_completion = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    organization_profile = relationship("OrganizationProfile", back_populates="migration_project", uselist=False)
    workload_profiles = relationship("WorkloadProfile", back_populates="migration_project")
    performance_requirements = relationship("PerformanceRequirements", back_populates="migration_project", uselist=False)
    compliance_requirements = relationship("ComplianceRequirements", back_populates="migration_project", uselist=False)
    budget_constraints = relationship("BudgetConstraints", back_populates="migration_project", uselist=False)
    technical_requirements = relationship("TechnicalRequirements", back_populates="migration_project", uselist=False)
    provider_evaluations = relationship("ProviderEvaluation", back_populates="migration_project")
    recommendation_report = relationship("RecommendationReport", back_populates="migration_project", uselist=False)
    migration_plan = relationship("MigrationPlan", back_populates="migration_project", uselist=False)
    
    # Indexes
    __table_args__ = (
        Index('ix_migration_projects_status', 'status'),
        Index('ix_migration_projects_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<MigrationProject(project_id='{self.project_id}', status='{self.status.value}')>"


class OrganizationProfile(Base):
    """Organization information for assessment"""
    __tablename__ = "organization_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    company_size = Column(Enum(CompanySize), nullable=False)
    industry = Column(String(100), nullable=False)
    current_infrastructure = Column(Enum(InfrastructureType), nullable=False)
    geographic_presence = Column(JSONB, default=[])  # List of regions/countries
    it_team_size = Column(Integer, nullable=False)
    cloud_experience_level = Column(Enum(ExperienceLevel), nullable=False)
    additional_context = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="organization_profile")
    
    def __repr__(self):
        return f"<OrganizationProfile(size='{self.company_size.value}', industry='{self.industry}')>"


class WorkloadProfile(Base):
    """Comprehensive workload information"""
    __tablename__ = "workload_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, index=True)
    workload_name = Column(String(255), nullable=False)
    application_type = Column(String(100), nullable=False)
    total_compute_cores = Column(Integer)
    total_memory_gb = Column(Integer)
    total_storage_tb = Column(Float)
    database_types = Column(JSONB, default=[])  # List of database types
    data_volume_tb = Column(Float)
    peak_transaction_rate = Column(Integer)
    workload_patterns = Column(JSONB, default={})  # Usage patterns, peak times, etc.
    dependencies = Column(JSONB, default=[])  # List of dependent workloads
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="workload_profiles")
    
    def __repr__(self):
        return f"<WorkloadProfile(name='{self.workload_name}', type='{self.application_type}')>"


class PerformanceRequirements(Base):
    """Performance and availability requirements"""
    __tablename__ = "performance_requirements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    latency_requirements = Column(JSONB, default={})  # Latency profiles by region/service
    availability_target = Column(Float, nullable=False)  # e.g., 99.99
    disaster_recovery_rto = Column(Integer)  # Recovery Time Objective in minutes
    disaster_recovery_rpo = Column(Integer)  # Recovery Point Objective in minutes
    geographic_distribution = Column(JSONB, default=[])  # Required regions
    peak_load_multiplier = Column(Float)
    additional_requirements = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="performance_requirements")
    
    @validates('availability_target')
    def validate_availability(self, key, value):
        """Validate availability target is between 0 and 100"""
        if not 0 <= value <= 100:
            raise ValueError("Availability target must be between 0 and 100")
        return value
    
    def __repr__(self):
        return f"<PerformanceRequirements(availability={self.availability_target}%)>"


class ComplianceRequirements(Base):
    """Regulatory and compliance needs"""
    __tablename__ = "compliance_requirements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    regulatory_frameworks = Column(JSONB, default=[])  # GDPR, HIPAA, SOC2, etc.
    data_residency_requirements = Column(JSONB, default=[])  # Countries/regions
    industry_certifications = Column(JSONB, default=[])  # Required certifications
    security_standards = Column(JSONB, default=[])  # Security standards
    audit_requirements = Column(JSONB, default={})
    additional_compliance = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="compliance_requirements")
    
    def __repr__(self):
        return f"<ComplianceRequirements(frameworks={len(self.regulatory_frameworks)})>"


class BudgetConstraints(Base):
    """Budget information"""
    __tablename__ = "budget_constraints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    current_monthly_cost = Column(Numeric(precision=12, scale=2))
    migration_budget = Column(Numeric(precision=12, scale=2), nullable=False)
    target_monthly_cost = Column(Numeric(precision=12, scale=2))
    cost_optimization_priority = Column(String(20))  # low, medium, high
    acceptable_cost_variance = Column(Float)  # Percentage
    currency = Column(String(3), default="USD")
    additional_constraints = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="budget_constraints")
    
    @validates('migration_budget')
    def validate_budget(self, key, value):
        """Validate budget is positive"""
        if value <= 0:
            raise ValueError("Migration budget must be positive")
        return value
    
    def __repr__(self):
        return f"<BudgetConstraints(migration_budget={self.migration_budget})>"


class TechnicalRequirements(Base):
    """Required cloud services and capabilities"""
    __tablename__ = "technical_requirements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    required_services = Column(JSONB, default=[])  # List of required cloud services
    ml_ai_requirements = Column(JSONB, default={})
    analytics_requirements = Column(JSONB, default={})
    container_orchestration = Column(Boolean, default=False)
    serverless_requirements = Column(Boolean, default=False)
    specialized_compute = Column(JSONB, default=[])  # GPU, HPC, etc.
    integration_requirements = Column(JSONB, default={})
    additional_technical = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="technical_requirements")
    
    def __repr__(self):
        return f"<TechnicalRequirements(services={len(self.required_services)})>"


# Recommendation Models

class ProviderEvaluation(Base):
    """Evaluation results for cloud providers"""
    __tablename__ = "provider_evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, index=True)
    provider_name = Column(String(50), nullable=False)  # AWS, GCP, Azure
    service_availability_score = Column(Float, nullable=False)
    pricing_score = Column(Float, nullable=False)
    compliance_score = Column(Float, nullable=False)
    technical_fit_score = Column(Float, nullable=False)
    migration_complexity_score = Column(Float, nullable=False)
    overall_score = Column(Float, nullable=False)
    strengths = Column(JSONB, default=[])
    weaknesses = Column(JSONB, default=[])
    detailed_analysis = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="provider_evaluations")
    
    # Indexes
    __table_args__ = (
        Index('ix_provider_evaluations_project_score', 'migration_project_id', 'overall_score'),
    )
    
    def __repr__(self):
        return f"<ProviderEvaluation(provider='{self.provider_name}', score={self.overall_score})>"


class RecommendationReport(Base):
    """Complete recommendation with justification"""
    __tablename__ = "recommendation_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    primary_recommendation = Column(String(50), nullable=False)  # AWS, GCP, or Azure
    confidence_score = Column(Float, nullable=False)
    key_differentiators = Column(JSONB, default=[])
    cost_comparison = Column(JSONB, default={})
    risk_assessment = Column(JSONB, default={})
    justification = Column(Text, nullable=False)
    scoring_weights = Column(JSONB, default={})  # Weights used for scoring
    alternative_recommendations = Column(JSONB, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="recommendation_report")
    
    @validates('confidence_score')
    def validate_confidence(self, key, value):
        """Validate confidence score is between 0 and 1"""
        if not 0 <= value <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        return value
    
    def __repr__(self):
        return f"<RecommendationReport(recommendation='{self.primary_recommendation}', confidence={self.confidence_score})>"


# Migration Planning Models

class MigrationPlan(Base):
    """Comprehensive migration plan"""
    __tablename__ = "migration_plans"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_id = Column(String(100), unique=True, nullable=False, index=True)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    target_provider = Column(String(50), nullable=False)
    total_duration_days = Column(Integer, nullable=False)
    estimated_cost = Column(Numeric(precision=12, scale=2), nullable=False)
    risk_level = Column(Enum(MigrationRiskLevel), nullable=False)
    dependencies_graph = Column(JSONB, default={})
    migration_waves = Column(JSONB, default=[])
    success_criteria = Column(JSONB, default=[])
    rollback_strategy = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_project = relationship("MigrationProject", back_populates="migration_plan")
    phases = relationship("MigrationPhase", back_populates="migration_plan", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MigrationPlan(plan_id='{self.plan_id}', provider='{self.target_provider}')>"


class MigrationPhase(Base):
    """Individual migration phase"""
    __tablename__ = "migration_phases"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phase_id = Column(String(100), nullable=False, index=True)
    migration_plan_id = Column(UUID(as_uuid=True), ForeignKey("migration_plans.id"), nullable=False, index=True)
    phase_name = Column(String(255), nullable=False)
    phase_order = Column(Integer, nullable=False)
    workloads = Column(JSONB, default=[])  # List of workload IDs
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    actual_start_date = Column(DateTime(timezone=True))
    actual_end_date = Column(DateTime(timezone=True))
    status = Column(Enum(PhaseStatus), nullable=False, default=PhaseStatus.NOT_STARTED)
    prerequisites = Column(JSONB, default=[])
    success_criteria = Column(JSONB, default=[])
    rollback_plan = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    migration_plan = relationship("MigrationPlan", back_populates="phases")
    
    # Indexes
    __table_args__ = (
        Index('ix_migration_phases_plan_order', 'migration_plan_id', 'phase_order'),
        Index('ix_migration_phases_status', 'status'),
    )
    
    def __repr__(self):
        return f"<MigrationPhase(phase_id='{self.phase_id}', name='{self.phase_name}', status='{self.status.value}')>"


# Resource Organization Models

class OrganizationalStructure(Base):
    """Organizational hierarchy definition"""
    __tablename__ = "organizational_structures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, index=True)
    structure_name = Column(String(255), nullable=False)
    teams = Column(JSONB, default=[])
    projects = Column(JSONB, default=[])
    environments = Column(JSONB, default=[])  # dev, staging, prod, etc.
    regions = Column(JSONB, default=[])
    cost_centers = Column(JSONB, default=[])
    custom_dimensions = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<OrganizationalStructure(name='{self.structure_name}')>"


class CategorizedResource(Base):
    """Resource with organizational categorization"""
    __tablename__ = "categorized_resources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, index=True)
    resource_id = Column(String(255), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False)
    resource_name = Column(String(255))
    provider = Column(String(50), nullable=False)
    team = Column(String(100))
    project = Column(String(100))
    environment = Column(String(50))
    region = Column(String(50))
    cost_center = Column(String(100))
    custom_attributes = Column(JSONB, default={})
    tags = Column(JSONB, default={})
    ownership_status = Column(Enum(OwnershipStatus), nullable=False, default=OwnershipStatus.UNASSIGNED)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('ix_categorized_resources_project_resource', 'migration_project_id', 'resource_id'),
        Index('ix_categorized_resources_team', 'team'),
        Index('ix_categorized_resources_project', 'project'),
        Index('ix_categorized_resources_ownership', 'ownership_status'),
    )
    
    def __repr__(self):
        return f"<CategorizedResource(resource_id='{self.resource_id}', team='{self.team}')>"


# Post-Migration Integration Models

class BaselineMetrics(Base):
    """Initial performance and cost baselines captured post-migration"""
    __tablename__ = "baseline_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, index=True)
    capture_date = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    total_monthly_cost = Column(Numeric(precision=12, scale=2), nullable=False)
    cost_by_service = Column(JSONB, default={})  # Service name -> cost
    cost_by_team = Column(JSONB, default={})  # Team -> cost
    cost_by_project = Column(JSONB, default={})  # Project -> cost
    cost_by_environment = Column(JSONB, default={})  # Environment -> cost
    resource_utilization = Column(JSONB, default={})  # Resource ID -> utilization metrics
    performance_metrics = Column(JSONB, default={})  # Service -> performance data
    resource_count = Column(Integer, nullable=False, default=0)
    resource_count_by_type = Column(JSONB, default={})  # Resource type -> count
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('ix_baseline_metrics_project_date', 'migration_project_id', 'capture_date'),
    )
    
    def __repr__(self):
        return f"<BaselineMetrics(project_id='{self.migration_project_id}', cost={self.total_monthly_cost})>"


class MigrationReport(Base):
    """Comprehensive migration completion report"""
    __tablename__ = "migration_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    migration_project_id = Column(UUID(as_uuid=True), ForeignKey("migration_projects.id"), nullable=False, unique=True)
    report_date = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    start_date = Column(DateTime(timezone=True), nullable=False)
    completion_date = Column(DateTime(timezone=True), nullable=False)
    actual_duration_days = Column(Integer, nullable=False)
    planned_duration_days = Column(Integer, nullable=False)
    total_cost = Column(Numeric(precision=12, scale=2), nullable=False)
    budgeted_cost = Column(Numeric(precision=12, scale=2), nullable=False)
    resources_migrated = Column(Integer, nullable=False)
    success_rate = Column(Float, nullable=False)  # Percentage
    lessons_learned = Column(JSONB, default=[])
    optimization_opportunities = Column(JSONB, default=[])
    cost_analysis = Column(JSONB, default={})
    timeline_analysis = Column(JSONB, default={})
    risk_incidents = Column(JSONB, default=[])
    recommendations = Column(JSONB, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    @validates('success_rate')
    def validate_success_rate(self, key, value):
        """Validate success rate is between 0 and 100"""
        if not 0 <= value <= 100:
            raise ValueError("Success rate must be between 0 and 100")
        return value
    
    def __repr__(self):
        return f"<MigrationReport(project_id='{self.migration_project_id}', success_rate={self.success_rate}%)>"
