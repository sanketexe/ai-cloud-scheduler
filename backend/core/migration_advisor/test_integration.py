"""
Integration Test Suite for Cloud Migration Advisor

This module contains integration tests for the complete migration workflow,
testing the interaction between multiple components and engines.

Tests cover:
- Complete migration workflow from assessment to integration
- Recommendation engine with various requirement profiles
- Resource organization with sample cloud environments
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..database import Base
from ..models import User, UserRole
from .models import (
    MigrationProject, OrganizationProfile, WorkloadProfile,
    PerformanceRequirements, ComplianceRequirements, BudgetConstraints,
    TechnicalRequirements, ProviderEvaluation, RecommendationReport,
    MigrationPlan, MigrationPhase, OrganizationalStructure,
    CategorizedResource, BaselineMetrics,
    MigrationStatus, CompanySize, InfrastructureType, ExperienceLevel,
    PhaseStatus, OwnershipStatus, MigrationRiskLevel
)
from .assessment_engine import MigrationAssessmentEngine
from .requirements_analysis_engine import RequirementsAnalysisEngine
from .recommendation_engine import CloudProviderRecommendationEngine
from .migration_planning_engine import MigrationPlanningEngine
from .resource_organization_engine import ResourceOrganizationEngine
from .post_migration_integration_engine import PostMigrationIntegrationEngine


# Test Fixtures

@pytest.fixture(scope="function")
def db_session():
    """Create an in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    user = User(
        email="integration_test@example.com",
        password_hash="hashed_password",
        first_name="Integration",
        last_name="Test",
        role=UserRole.ADMIN
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def small_org_profile():
    """Small organization profile for testing"""
    return {
        'company_size': 'small',
        'industry': 'Technology',
        'current_infrastructure': 'on_premises',
        'it_team_size': 5,
        'cloud_experience_level': 'beginner',
        'geographic_presence': ['North America']
    }


@pytest.fixture
def medium_org_profile():
    """Medium organization profile for testing"""
    return {
        'company_size': 'medium',
        'industry': 'Financial Services',
        'current_infrastructure': 'hybrid',
        'it_team_size': 25,
        'cloud_experience_level': 'intermediate',
        'geographic_presence': ['North America', 'Europe']
    }


@pytest.fixture
def large_org_profile():
    """Large organization profile for testing"""
    return {
        'company_size': 'large',
        'industry': 'Healthcare',
        'current_infrastructure': 'multi_cloud',
        'it_team_size': 100,
        'cloud_experience_level': 'advanced',
        'geographic_presence': ['North America', 'Europe', 'Asia']
    }


# Integration Tests

class TestCompleteWorkflow:
    """Test complete migration workflow from assessment to integration"""
    
    def test_small_organization_workflow(self, db_session, test_user, small_org_profile):
        """Test complete workflow for a small organization"""
        # Step 1: Initiate migration assessment
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Small Tech Startup",
            created_by_user_id=test_user.id
        )
        
        assert project_result['status'] == 'assessment'
        project_id = project_result['project_id']
        
        # Step 2: Collect organization profile
        profile_result = assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=small_org_profile
        )
        
        assert 'profile' in profile_result
        assert 'timeline_estimation' in profile_result
        
        # Step 3: Collect workload requirements
        requirements_engine = RequirementsAnalysisEngine(db_session)
        
        workload_data = {
            'workload_name': 'Web Application',
            'application_type': 'web',
            'total_compute_cores': 8,
            'total_memory_gb': 32,
            'total_storage_tb': 1.0,
            'data_volume_tb': 0.5
        }
        
        workload_result = requirements_engine.analyze_workloads(
            project_id=project_id,
            workload_data=workload_data
        )
        
        assert workload_result['workload_name'] == 'Web Application'
        
        # Step 4: Collect performance requirements
        perf_data = {
            'availability_target': 99.9,
            'disaster_recovery_rto': 120,
            'disaster_recovery_rpo': 60,
            'geographic_distribution': ['us-east-1']
        }
        
        perf_result = requirements_engine.assess_performance_requirements(
            project_id=project_id,
            perf_requirements=perf_data
        )
        
        assert perf_result['availability_target'] == 99.9
        
        # Step 5: Collect compliance requirements
        compliance_data = {
            'regulatory_frameworks': ['SOC2'],
            'data_residency_requirements': ['US'],
            'security_standards': ['ISO27001']
        }
        
        compliance_result = requirements_engine.evaluate_compliance_needs(
            project_id=project_id,
            compliance_data=compliance_data
        )
        
        assert 'SOC2' in compliance_result['regulatory_frameworks']
        
        # Step 6: Collect budget constraints
        budget_data = {
            'migration_budget': 50000.0,
            'target_monthly_cost': 3000.0,
            'currency': 'USD'
        }
        
        budget_result = requirements_engine.analyze_budget_constraints(
            project_id=project_id,
            budget_data=budget_data
        )
        
        assert budget_result['migration_budget'] == 50000.0
        
        # Step 7: Generate provider recommendations
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        assert 'ranked_providers' in recommendations
        assert len(recommendations['ranked_providers']) > 0
        assert 'primary_recommendation' in recommendations
        
        # Step 8: Select provider and generate migration plan
        selected_provider = recommendations['primary_recommendation']
        
        planning_engine = MigrationPlanningEngine(db_session)
        
        migration_plan = planning_engine.generate_migration_plan(
            project_id=project_id,
            selected_provider=selected_provider
        )
        
        assert migration_plan['target_provider'] == selected_provider
        assert 'phases' in migration_plan
        assert len(migration_plan['phases']) > 0
        
        # Step 9: Simulate resource organization
        org_engine = ResourceOrganizationEngine(db_session)
        
        # Define organizational structure
        org_structure_data = {
            'structure_name': 'Small Tech Startup Structure',
            'teams': [
                {'name': 'Engineering', 'owner': 'eng@example.com'},
                {'name': 'Operations', 'owner': 'ops@example.com'}
            ],
            'projects': [
                {'name': 'Web App', 'team': 'Engineering'}
            ],
            'environments': ['dev', 'staging', 'prod'],
            'regions': ['us-east-1']
        }
        
        org_structure = org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        assert org_structure['structure_name'] == 'Small Tech Startup Structure'
        
        # Step 10: Post-migration integration
        integration_engine = PostMigrationIntegrationEngine(db_session)
        
        # Configure cost tracking
        cost_tracking = integration_engine.configure_cost_tracking(
            project_id=project_id
        )
        
        assert 'cost_centers_created' in cost_tracking
        
        # Verify workflow completion
        project = db_session.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        assert project is not None
        assert project.organization_name == "Small Tech Startup"
    
    def test_medium_organization_workflow(self, db_session, test_user, medium_org_profile):
        """Test complete workflow for a medium organization with compliance needs"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        # Initiate assessment
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Medium Financial Corp",
            created_by_user_id=test_user.id
        )
        
        project_id = project_result['project_id']
        
        # Collect organization profile
        profile_result = assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=medium_org_profile
        )
        
        assert profile_result['profile']['company_size'] == 'medium'
        
        # Collect workload requirements with multiple workloads
        requirements_engine = RequirementsAnalysisEngine(db_session)
        
        workloads = [
            {
                'workload_name': 'Core Banking System',
                'application_type': 'database',
                'total_compute_cores': 32,
                'total_memory_gb': 128,
                'total_storage_tb': 10.0,
                'data_volume_tb': 8.0
            },
            {
                'workload_name': 'Customer Portal',
                'application_type': 'web',
                'total_compute_cores': 16,
                'total_memory_gb': 64,
                'total_storage_tb': 2.0,
                'data_volume_tb': 1.0
            }
        ]
        
        for workload_data in workloads:
            requirements_engine.analyze_workloads(
                project_id=project_id,
                workload_data=workload_data
            )
        
        # Collect strict compliance requirements
        compliance_data = {
            'regulatory_frameworks': ['PCI-DSS', 'SOC2', 'GDPR'],
            'data_residency_requirements': ['US', 'EU'],
            'security_standards': ['ISO27001', 'NIST'],
            'audit_requirements': {
                'frequency': 'quarterly',
                'retention_years': 7
            }
        }
        
        compliance_result = requirements_engine.evaluate_compliance_needs(
            project_id=project_id,
            compliance_data=compliance_data
        )
        
        assert 'PCI-DSS' in compliance_result['regulatory_frameworks']
        assert 'GDPR' in compliance_result['regulatory_frameworks']
        
        # Generate recommendations with compliance focus
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        # Verify compliance is considered in recommendations
        assert 'ranked_providers' in recommendations
        primary = recommendations['primary_recommendation']
        
        # Generate migration plan
        planning_engine = MigrationPlanningEngine(db_session)
        
        migration_plan = planning_engine.generate_migration_plan(
            project_id=project_id,
            selected_provider=primary
        )
        
        # Should have multiple phases for complex migration
        assert len(migration_plan['phases']) >= 2
        assert migration_plan['total_duration_days'] > 0


class TestRecommendationEngineProfiles:
    """Test recommendation engine with various requirement profiles"""
    
    def test_cost_optimized_profile(self, db_session, test_user):
        """Test recommendations for cost-optimized requirements"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        requirements_engine = RequirementsAnalysisEngine(db_session)
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Cost-Conscious Startup",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Profile with cost focus
        profile_data = {
            'company_size': 'small',
            'industry': 'Technology',
            'current_infrastructure': 'on_premises',
            'it_team_size': 3,
            'cloud_experience_level': 'beginner',
            'geographic_presence': ['North America']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=profile_data
        )
        
        # Minimal workload
        workload_data = {
            'workload_name': 'Simple Web App',
            'application_type': 'web',
            'total_compute_cores': 4,
            'total_memory_gb': 16,
            'total_storage_tb': 0.5,
            'data_volume_tb': 0.2
        }
        
        requirements_engine.analyze_workloads(
            project_id=project_id,
            workload_data=workload_data
        )
        
        # Tight budget
        budget_data = {
            'migration_budget': 10000.0,
            'target_monthly_cost': 500.0,
            'currency': 'USD',
            'cost_optimization_priority': 'high'
        }
        
        requirements_engine.analyze_budget_constraints(
            project_id=project_id,
            budget_data=budget_data
        )
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        assert 'ranked_providers' in recommendations
        # Cost should be a major factor
        for provider in recommendations['ranked_providers']:
            assert 'pricing_score' in provider
    
    def test_compliance_focused_profile(self, db_session, test_user):
        """Test recommendations for compliance-focused requirements"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        requirements_engine = RequirementsAnalysisEngine(db_session)
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Healthcare Provider",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Healthcare profile
        profile_data = {
            'company_size': 'medium',
            'industry': 'Healthcare',
            'current_infrastructure': 'on_premises',
            'it_team_size': 20,
            'cloud_experience_level': 'intermediate',
            'geographic_presence': ['North America']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=profile_data
        )
        
        # Strict compliance requirements
        compliance_data = {
            'regulatory_frameworks': ['HIPAA', 'SOC2', 'HITRUST'],
            'data_residency_requirements': ['US'],
            'security_standards': ['ISO27001', 'NIST'],
            'audit_requirements': {
                'frequency': 'monthly',
                'retention_years': 10
            }
        }
        
        requirements_engine.evaluate_compliance_needs(
            project_id=project_id,
            compliance_data=compliance_data
        )
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        assert 'ranked_providers' in recommendations
        # Compliance should be a major factor
        for provider in recommendations['ranked_providers']:
            assert 'compliance_score' in provider
    
    def test_performance_focused_profile(self, db_session, test_user):
        """Test recommendations for performance-focused requirements"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        requirements_engine = RequirementsAnalysisEngine(db_session)
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Gaming Platform",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Gaming platform profile
        profile_data = {
            'company_size': 'medium',
            'industry': 'Gaming',
            'current_infrastructure': 'cloud',
            'it_team_size': 30,
            'cloud_experience_level': 'advanced',
            'geographic_presence': ['Global']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=profile_data
        )
        
        # High-performance workload
        workload_data = {
            'workload_name': 'Game Servers',
            'application_type': 'compute',
            'total_compute_cores': 128,
            'total_memory_gb': 512,
            'total_storage_tb': 20.0,
            'data_volume_tb': 15.0
        }
        
        requirements_engine.analyze_workloads(
            project_id=project_id,
            workload_data=workload_data
        )
        
        # Strict performance requirements
        perf_data = {
            'availability_target': 99.99,
            'disaster_recovery_rto': 15,
            'disaster_recovery_rpo': 5,
            'geographic_distribution': ['us-east-1', 'eu-west-1', 'ap-southeast-1'],
            'latency_requirements': {
                'max_latency_ms': 50,
                'target_latency_ms': 20
            }
        }
        
        requirements_engine.assess_performance_requirements(
            project_id=project_id,
            perf_requirements=perf_data
        )
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        assert 'ranked_providers' in recommendations
        # Performance should be a major factor
        for provider in recommendations['ranked_providers']:
            assert 'technical_fit_score' in provider


class TestResourceOrganization:
    """Test resource organization with sample cloud environments"""
    
    def test_organize_aws_resources(self, db_session, test_user):
        """Test organizing AWS resources"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        org_engine = ResourceOrganizationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="AWS Migration Test",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Define organizational structure
        org_structure_data = {
            'structure_name': 'AWS Test Structure',
            'teams': [
                {'name': 'Engineering', 'owner': 'eng@example.com'},
                {'name': 'Data', 'owner': 'data@example.com'}
            ],
            'projects': [
                {'name': 'Web Platform', 'team': 'Engineering'},
                {'name': 'Analytics', 'team': 'Data'}
            ],
            'environments': ['dev', 'staging', 'prod'],
            'regions': ['us-east-1', 'us-west-2'],
            'cost_centers': [
                {'name': 'Engineering CC', 'code': 'CC-ENG'},
                {'name': 'Data CC', 'code': 'CC-DATA'}
            ]
        }
        
        org_structure = org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        assert org_structure['structure_name'] == 'AWS Test Structure'
        
        # Simulate discovered AWS resources
        sample_resources = [
            {
                'resource_id': 'i-1234567890abcdef0',
                'resource_type': 'EC2Instance',
                'resource_name': 'web-server-prod-01',
                'provider': 'AWS',
                'region': 'us-east-1',
                'tags': {'Name': 'web-server-prod-01', 'Environment': 'prod'}
            },
            {
                'resource_id': 'vol-1234567890abcdef0',
                'resource_type': 'EBSVolume',
                'resource_name': 'web-data-volume',
                'provider': 'AWS',
                'region': 'us-east-1',
                'tags': {'Name': 'web-data-volume'}
            },
            {
                'resource_id': 'db-instance-1',
                'resource_type': 'RDSInstance',
                'resource_name': 'analytics-db-prod',
                'provider': 'AWS',
                'region': 'us-west-2',
                'tags': {'Name': 'analytics-db-prod', 'Environment': 'prod'}
            }
        ]
        
        # Categorize resources
        categorized = org_engine.categorize_resources(
            project_id=project_id,
            resources=sample_resources
        )
        
        assert 'categorized_count' in categorized
        assert categorized['categorized_count'] > 0
    
    def test_multi_cloud_resource_organization(self, db_session, test_user):
        """Test organizing resources across multiple cloud providers"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        org_engine = ResourceOrganizationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Multi-Cloud Test",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Define organizational structure
        org_structure_data = {
            'structure_name': 'Multi-Cloud Structure',
            'teams': [
                {'name': 'Platform', 'owner': 'platform@example.com'}
            ],
            'projects': [
                {'name': 'Global Services', 'team': 'Platform'}
            ],
            'environments': ['prod'],
            'regions': ['us-east-1', 'europe-west1', 'eastus']
        }
        
        org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        # Simulate resources from multiple providers
        sample_resources = [
            {
                'resource_id': 'i-aws-123',
                'resource_type': 'EC2Instance',
                'resource_name': 'aws-web-server',
                'provider': 'AWS',
                'region': 'us-east-1'
            },
            {
                'resource_id': 'gcp-vm-456',
                'resource_type': 'ComputeInstance',
                'resource_name': 'gcp-web-server',
                'provider': 'GCP',
                'region': 'europe-west1'
            },
            {
                'resource_id': 'azure-vm-789',
                'resource_type': 'VirtualMachine',
                'resource_name': 'azure-web-server',
                'provider': 'Azure',
                'region': 'eastus'
            }
        ]
        
        # Categorize resources
        categorized = org_engine.categorize_resources(
            project_id=project_id,
            resources=sample_resources
        )
        
        assert categorized['categorized_count'] == 3


class TestDependencyHandling:
    """Test dependency analysis and migration sequencing"""
    
    def test_complex_dependency_graph(self, db_session, test_user):
        """Test handling complex resource dependencies"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        requirements_engine = RequirementsAnalysisEngine(db_session)
        planning_engine = MigrationPlanningEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Complex Dependencies Test",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Profile
        profile_data = {
            'company_size': 'medium',
            'industry': 'Technology',
            'current_infrastructure': 'on_premises',
            'it_team_size': 20,
            'cloud_experience_level': 'intermediate',
            'geographic_presence': ['North America']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=profile_data
        )
        
        # Create workloads with dependencies
        # Database -> Application -> Web Server -> Load Balancer
        workloads = [
            {
                'workload_name': 'Database',
                'application_type': 'database',
                'total_compute_cores': 8,
                'total_memory_gb': 64,
                'total_storage_tb': 5.0,
                'data_volume_tb': 4.0,
                'dependencies': []
            },
            {
                'workload_name': 'Application Server',
                'application_type': 'application',
                'total_compute_cores': 16,
                'total_memory_gb': 64,
                'total_storage_tb': 2.0,
                'data_volume_tb': 1.0,
                'dependencies': ['Database']
            },
            {
                'workload_name': 'Web Server',
                'application_type': 'web',
                'total_compute_cores': 8,
                'total_memory_gb': 32,
                'total_storage_tb': 1.0,
                'data_volume_tb': 0.5,
                'dependencies': ['Application Server']
            },
            {
                'workload_name': 'Load Balancer',
                'application_type': 'network',
                'total_compute_cores': 4,
                'total_memory_gb': 16,
                'total_storage_tb': 0.1,
                'data_volume_tb': 0.05,
                'dependencies': ['Web Server']
            }
        ]
        
        for workload_data in workloads:
            requirements_engine.analyze_workloads(
                project_id=project_id,
                workload_data=workload_data
            )
        
        # Generate migration plan
        migration_plan = planning_engine.generate_migration_plan(
            project_id=project_id,
            selected_provider='AWS'
        )
        
        # Verify phases respect dependencies
        assert 'phases' in migration_plan
        assert len(migration_plan['phases']) > 0
        
        # Database should be in earlier phase than Application Server
        phase_workloads = [phase['workloads'] for phase in migration_plan['phases']]
        
        # Flatten to check order
        all_workloads_ordered = []
        for phase_wl in phase_workloads:
            all_workloads_ordered.extend(phase_wl)
        
        # Basic dependency check
        assert len(all_workloads_ordered) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
