"""
End-to-End Test Scenarios for Cloud Migration Advisor

This module contains comprehensive end-to-end test scenarios that simulate
real-world migration workflows for organizations of different sizes.

Test scenarios:
- Small organization: Simple migration with minimal complexity
- Medium organization: Multi-workload migration with compliance requirements
- Large organization: Complex enterprise migration with multiple dependencies
- FinOps integration and handoff validation
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
    TechnicalRequirements, MigrationStatus, CompanySize,
    InfrastructureType, ExperienceLevel
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
        email="e2e_test@example.com",
        password_hash="hashed_password",
        first_name="E2E",
        last_name="Test",
        role=UserRole.ADMIN
    )
    db_session.add(user)
    db_session.commit()
    return user


# End-to-End Test Scenarios

class TestSmallOrganizationE2E:
    """Complete end-to-end test for small organization migration"""
    
    def test_small_tech_startup_complete_flow(self, db_session, test_user):
        """
        Scenario: Small tech startup (10 employees) migrating simple web app
        from on-premises to cloud for the first time.
        
        Requirements:
        - Single web application with database
        - Limited budget ($20k migration, $2k/month operational)
        - Basic compliance (SOC2)
        - US-only operations
        - Minimal cloud experience
        """
        # Phase 1: Assessment
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="TechStart Inc",
            created_by_user_id=test_user.id
        )
        
        assert project_result['status'] == 'assessment'
        project_id = project_result['project_id']
        
        # Collect organization profile
        org_profile = {
            'company_size': 'small',
            'industry': 'Technology',
            'current_infrastructure': 'on_premises',
            'it_team_size': 2,
            'cloud_experience_level': 'none',
            'geographic_presence': ['North America']
        }
        
        profile_result = assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=org_profile
        )
        
        assert 'timeline_estimation' in profile_result
        estimated_days = profile_result['timeline_estimation']['estimated_days']
        assert estimated_days > 0
        
        # Phase 2: Requirements Analysis
        requirements_engine = RequirementsAnalysisEngine(db_session)
        
        # Single workload
        workload_data = {
            'workload_name': 'Web Application + Database',
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
        
        # Basic performance requirements
        perf_data = {
            'availability_target': 99.5,
            'disaster_recovery_rto': 240,
            'disaster_recovery_rpo': 120,
            'geographic_distribution': ['us-east-1']
        }
        
        requirements_engine.assess_performance_requirements(
            project_id=project_id,
            perf_requirements=perf_data
        )
        
        # Basic compliance
        compliance_data = {
            'regulatory_frameworks': ['SOC2'],
            'data_residency_requirements': ['US'],
            'security_standards': ['ISO27001']
        }
        
        requirements_engine.evaluate_compliance_needs(
            project_id=project_id,
            compliance_data=compliance_data
        )
        
        # Tight budget
        budget_data = {
            'migration_budget': 20000.0,
            'target_monthly_cost': 2000.0,
            'currency': 'USD',
            'cost_optimization_priority': 'high'
        }
        
        requirements_engine.analyze_budget_constraints(
            project_id=project_id,
            budget_data=budget_data
        )
        
        # Phase 3: Provider Recommendation
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        assert 'ranked_providers' in recommendations
        assert 'primary_recommendation' in recommendations
        assert len(recommendations['ranked_providers']) >= 1
        
        selected_provider = recommendations['primary_recommendation']
        
        # Phase 4: Migration Planning
        planning_engine = MigrationPlanningEngine(db_session)
        
        migration_plan = planning_engine.generate_migration_plan(
            project_id=project_id,
            selected_provider=selected_provider
        )
        
        assert migration_plan['target_provider'] == selected_provider
        assert 'phases' in migration_plan
        assert migration_plan['total_duration_days'] > 0
        
        # Phase 5: Resource Organization
        org_engine = ResourceOrganizationEngine(db_session)
        
        org_structure_data = {
            'structure_name': 'TechStart Structure',
            'teams': [
                {'name': 'Engineering', 'owner': 'eng@techstart.com'}
            ],
            'projects': [
                {'name': 'Main App', 'team': 'Engineering'}
            ],
            'environments': ['prod'],
            'regions': ['us-east-1']
        }
        
        org_structure = org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        assert org_structure['structure_name'] == 'TechStart Structure'
        
        # Phase 6: Post-Migration Integration
        integration_engine = PostMigrationIntegrationEngine(db_session)
        
        cost_tracking = integration_engine.configure_cost_tracking(project_id)
        assert 'cost_centers_created' in cost_tracking
        
        finops_integration = integration_engine.integrate_finops_capabilities(
            project_id=project_id,
            provider=selected_provider
        )
        assert 'features_enabled' in finops_integration
        
        # Verify complete workflow
        project = db_session.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        assert project is not None
        assert project.organization_name == "TechStart Inc"


class TestMediumOrganizationE2E:
    """Complete end-to-end test for medium organization migration"""
    
    def test_financial_services_company_complete_flow(self, db_session, test_user):
        """
        Scenario: Medium financial services company (100 employees) migrating
        multiple applications with strict compliance requirements.
        
        Requirements:
        - Multiple workloads (core banking, customer portal, analytics)
        - Moderate budget ($200k migration, $20k/month operational)
        - Strict compliance (PCI-DSS, SOC2, GDPR)
        - Multi-region (US, EU)
        - Intermediate cloud experience
        """
        # Phase 1: Assessment
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="SecureBank Corp",
            created_by_user_id=test_user.id
        )
        
        project_id = project_result['project_id']
        
        org_profile = {
            'company_size': 'medium',
            'industry': 'Financial Services',
            'current_infrastructure': 'hybrid',
            'it_team_size': 25,
            'cloud_experience_level': 'intermediate',
            'geographic_presence': ['North America', 'Europe']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=org_profile
        )
        
        # Phase 2: Requirements Analysis - Multiple Workloads
        requirements_engine = RequirementsAnalysisEngine(db_session)
        
        workloads = [
            {
                'workload_name': 'Core Banking System',
                'application_type': 'database',
                'total_compute_cores': 32,
                'total_memory_gb': 128,
                'total_storage_tb': 10.0,
                'data_volume_tb': 8.0,
                'dependencies': []
            },
            {
                'workload_name': 'Customer Portal',
                'application_type': 'web',
                'total_compute_cores': 16,
                'total_memory_gb': 64,
                'total_storage_tb': 2.0,
                'data_volume_tb': 1.0,
                'dependencies': ['Core Banking System']
            },
            {
                'workload_name': 'Analytics Platform',
                'application_type': 'analytics',
                'total_compute_cores': 24,
                'total_memory_gb': 96,
                'total_storage_tb': 15.0,
                'data_volume_tb': 12.0,
                'dependencies': ['Core Banking System']
            }
        ]
        
        for workload_data in workloads:
            requirements_engine.analyze_workloads(
                project_id=project_id,
                workload_data=workload_data
            )
        
        # Strict performance requirements
        perf_data = {
            'availability_target': 99.95,
            'disaster_recovery_rto': 60,
            'disaster_recovery_rpo': 30,
            'geographic_distribution': ['us-east-1', 'eu-west-1']
        }
        
        requirements_engine.assess_performance_requirements(
            project_id=project_id,
            perf_requirements=perf_data
        )
        
        # Strict compliance requirements
        compliance_data = {
            'regulatory_frameworks': ['PCI-DSS', 'SOC2', 'GDPR'],
            'data_residency_requirements': ['US', 'EU'],
            'security_standards': ['ISO27001', 'NIST'],
            'audit_requirements': {
                'frequency': 'quarterly',
                'retention_years': 7
            }
        }
        
        requirements_engine.evaluate_compliance_needs(
            project_id=project_id,
            compliance_data=compliance_data
        )
        
        # Moderate budget
        budget_data = {
            'migration_budget': 200000.0,
            'target_monthly_cost': 20000.0,
            'currency': 'USD',
            'cost_optimization_priority': 'medium'
        }
        
        requirements_engine.analyze_budget_constraints(
            project_id=project_id,
            budget_data=budget_data
        )
        
        # Phase 3: Provider Recommendation
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        assert 'ranked_providers' in recommendations
        assert len(recommendations['ranked_providers']) >= 1
        
        # Verify compliance is considered
        for provider in recommendations['ranked_providers']:
            assert 'compliance_score' in provider
            assert provider['compliance_score'] > 0
        
        selected_provider = recommendations['primary_recommendation']
        
        # Phase 4: Migration Planning with Dependencies
        planning_engine = MigrationPlanningEngine(db_session)
        
        migration_plan = planning_engine.generate_migration_plan(
            project_id=project_id,
            selected_provider=selected_provider
        )
        
        # Should have multiple phases due to dependencies
        assert len(migration_plan['phases']) >= 2
        assert migration_plan['total_duration_days'] > 0
        
        # Phase 5: Resource Organization with Multiple Teams
        org_engine = ResourceOrganizationEngine(db_session)
        
        org_structure_data = {
            'structure_name': 'SecureBank Structure',
            'teams': [
                {'name': 'Core Banking', 'owner': 'banking@securebank.com'},
                {'name': 'Digital', 'owner': 'digital@securebank.com'},
                {'name': 'Analytics', 'owner': 'analytics@securebank.com'}
            ],
            'projects': [
                {'name': 'Banking Platform', 'team': 'Core Banking'},
                {'name': 'Customer Portal', 'team': 'Digital'},
                {'name': 'Data Analytics', 'team': 'Analytics'}
            ],
            'environments': ['dev', 'staging', 'prod'],
            'regions': ['us-east-1', 'eu-west-1'],
            'cost_centers': [
                {'name': 'Banking CC', 'code': 'CC-BANK'},
                {'name': 'Digital CC', 'code': 'CC-DIGI'},
                {'name': 'Analytics CC', 'code': 'CC-ANLY'}
            ]
        }
        
        org_structure = org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        assert len(org_structure['teams']) == 3
        assert len(org_structure['projects']) == 3
        
        # Phase 6: Post-Migration Integration with FinOps
        integration_engine = PostMigrationIntegrationEngine(db_session)
        
        cost_tracking = integration_engine.configure_cost_tracking(project_id)
        assert len(cost_tracking['cost_centers_created']) == 3
        
        finops_integration = integration_engine.integrate_finops_capabilities(
            project_id=project_id,
            provider=selected_provider
        )
        assert 'features_enabled' in finops_integration
        
        # Capture baselines
        sample_resources = [
            {'resource_id': 'r1', 'resource_type': 'compute_instance'},
            {'resource_id': 'r2', 'resource_type': 'database'},
            {'resource_id': 'r3', 'resource_type': 'storage_bucket'}
        ]
        
        baseline = integration_engine.capture_baselines(
            project_id=project_id,
            resources=sample_resources
        )
        
        assert baseline['resource_count'] == 3
        assert baseline['total_monthly_cost'] > 0
        
        # Generate final report
        report = integration_engine.generate_migration_report(project_id)
        
        assert report['project_id'] == project_id
        assert 'resources_migrated' in report


class TestLargeOrganizationE2E:
    """Complete end-to-end test for large enterprise migration"""
    
    def test_enterprise_healthcare_complete_flow(self, db_session, test_user):
        """
        Scenario: Large healthcare enterprise (1000+ employees) migrating
        complex infrastructure with strict regulatory requirements.
        
        Requirements:
        - Many workloads with complex dependencies
        - Large budget ($1M migration, $100k/month operational)
        - Very strict compliance (HIPAA, HITRUST, SOC2, GDPR)
        - Global operations (US, EU, Asia)
        - Advanced cloud experience
        """
        # Phase 1: Assessment
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="GlobalHealth Enterprise",
            created_by_user_id=test_user.id
        )
        
        project_id = project_result['project_id']
        
        org_profile = {
            'company_size': 'enterprise',
            'industry': 'Healthcare',
            'current_infrastructure': 'hybrid',
            'it_team_size': 150,
            'cloud_experience_level': 'advanced',
            'geographic_presence': ['North America', 'Europe', 'Asia']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=org_profile
        )
        
        # Phase 2: Requirements Analysis - Complex Workloads
        requirements_engine = RequirementsAnalysisEngine(db_session)
        
        # Multiple complex workloads
        workloads = [
            {
                'workload_name': 'Patient Records System',
                'application_type': 'database',
                'total_compute_cores': 64,
                'total_memory_gb': 512,
                'total_storage_tb': 50.0,
                'data_volume_tb': 40.0,
                'dependencies': []
            },
            {
                'workload_name': 'Clinical Applications',
                'application_type': 'application',
                'total_compute_cores': 48,
                'total_memory_gb': 256,
                'total_storage_tb': 10.0,
                'data_volume_tb': 5.0,
                'dependencies': ['Patient Records System']
            },
            {
                'workload_name': 'Patient Portal',
                'application_type': 'web',
                'total_compute_cores': 32,
                'total_memory_gb': 128,
                'total_storage_tb': 5.0,
                'data_volume_tb': 2.0,
                'dependencies': ['Patient Records System', 'Clinical Applications']
            },
            {
                'workload_name': 'Medical Imaging',
                'application_type': 'storage',
                'total_compute_cores': 24,
                'total_memory_gb': 96,
                'total_storage_tb': 100.0,
                'data_volume_tb': 80.0,
                'dependencies': ['Patient Records System']
            }
        ]
        
        for workload_data in workloads:
            requirements_engine.analyze_workloads(
                project_id=project_id,
                workload_data=workload_data
            )
        
        # Very strict performance requirements
        perf_data = {
            'availability_target': 99.99,
            'disaster_recovery_rto': 15,
            'disaster_recovery_rpo': 5,
            'geographic_distribution': ['us-east-1', 'eu-west-1', 'ap-southeast-1']
        }
        
        requirements_engine.assess_performance_requirements(
            project_id=project_id,
            perf_requirements=perf_data
        )
        
        # Very strict compliance requirements
        compliance_data = {
            'regulatory_frameworks': ['HIPAA', 'HITRUST', 'SOC2', 'GDPR'],
            'data_residency_requirements': ['US', 'EU', 'Asia'],
            'security_standards': ['ISO27001', 'NIST', 'FedRAMP'],
            'audit_requirements': {
                'frequency': 'monthly',
                'retention_years': 10
            }
        }
        
        requirements_engine.evaluate_compliance_needs(
            project_id=project_id,
            compliance_data=compliance_data
        )
        
        # Large budget
        budget_data = {
            'migration_budget': 1000000.0,
            'target_monthly_cost': 100000.0,
            'currency': 'USD',
            'cost_optimization_priority': 'low'
        }
        
        requirements_engine.analyze_budget_constraints(
            project_id=project_id,
            budget_data=budget_data
        )
        
        # Phase 3: Provider Recommendation
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        recommendations = recommendation_engine.generate_recommendations(project_id)
        
        assert 'ranked_providers' in recommendations
        
        # All providers should be evaluated
        assert len(recommendations['ranked_providers']) >= 1
        
        selected_provider = recommendations['primary_recommendation']
        
        # Phase 4: Complex Migration Planning
        planning_engine = MigrationPlanningEngine(db_session)
        
        migration_plan = planning_engine.generate_migration_plan(
            project_id=project_id,
            selected_provider=selected_provider
        )
        
        # Should have multiple phases due to complex dependencies
        assert len(migration_plan['phases']) >= 3
        assert migration_plan['total_duration_days'] > 30
        
        # Phase 5: Complex Resource Organization
        org_engine = ResourceOrganizationEngine(db_session)
        
        org_structure_data = {
            'structure_name': 'GlobalHealth Structure',
            'teams': [
                {'name': 'Clinical Systems', 'owner': 'clinical@globalhealth.com'},
                {'name': 'Patient Services', 'owner': 'patient@globalhealth.com'},
                {'name': 'Medical Imaging', 'owner': 'imaging@globalhealth.com'},
                {'name': 'IT Operations', 'owner': 'ops@globalhealth.com'}
            ],
            'projects': [
                {'name': 'EHR Platform', 'team': 'Clinical Systems'},
                {'name': 'Patient Portal', 'team': 'Patient Services'},
                {'name': 'PACS System', 'team': 'Medical Imaging'}
            ],
            'environments': ['dev', 'test', 'staging', 'prod'],
            'regions': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
            'cost_centers': [
                {'name': 'Clinical CC', 'code': 'CC-CLIN'},
                {'name': 'Patient CC', 'code': 'CC-PATI'},
                {'name': 'Imaging CC', 'code': 'CC-IMAG'},
                {'name': 'IT CC', 'code': 'CC-ITOP'}
            ]
        }
        
        org_structure = org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        assert len(org_structure['teams']) == 4
        assert len(org_structure['projects']) == 3
        assert len(org_structure['cost_centers']) == 4
        
        # Phase 6: Comprehensive Post-Migration Integration
        integration_engine = PostMigrationIntegrationEngine(db_session)
        
        cost_tracking = integration_engine.configure_cost_tracking(project_id)
        assert len(cost_tracking['cost_centers_created']) == 4
        
        finops_integration = integration_engine.integrate_finops_capabilities(
            project_id=project_id,
            provider=selected_provider
        )
        
        # Verify all FinOps features are enabled
        feature_names = [f['feature'] for f in finops_integration['features_enabled']]
        assert 'waste_detection' in feature_names
        assert 'ri_optimization' in feature_names
        assert 'cost_anomaly_detection' in feature_names
        
        # Generate comprehensive report
        report = integration_engine.generate_migration_report(project_id)
        
        assert report['project_id'] == project_id
        assert 'total_cost' in report or 'estimated_cost' in report


class TestFinOpsIntegrationValidation:
    """Test FinOps integration and handoff validation"""
    
    def test_finops_handoff_complete_workflow(self, db_session, test_user):
        """
        Test complete FinOps integration and handoff workflow
        """
        # Setup migration project
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="FinOps Integration Test",
            created_by_user_id=test_user.id
        )
        
        project_id = project_result['project_id']
        
        # Quick setup
        org_profile = {
            'company_size': 'medium',
            'industry': 'Technology',
            'current_infrastructure': 'on_premises',
            'it_team_size': 20,
            'cloud_experience_level': 'intermediate',
            'geographic_presence': ['North America']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=org_profile
        )
        
        # Define organizational structure
        org_engine = ResourceOrganizationEngine(db_session)
        
        org_structure_data = {
            'structure_name': 'FinOps Test Structure',
            'teams': [
                {'name': 'Engineering', 'owner': 'eng@test.com'},
                {'name': 'Operations', 'owner': 'ops@test.com'}
            ],
            'projects': [
                {'name': 'Project A', 'team': 'Engineering'},
                {'name': 'Project B', 'team': 'Operations'}
            ],
            'environments': ['dev', 'prod'],
            'regions': ['us-east-1'],
            'cost_centers': [
                {'name': 'Engineering CC', 'code': 'CC-ENG'},
                {'name': 'Operations CC', 'code': 'CC-OPS'}
            ]
        }
        
        org_structure = org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        # Test FinOps Integration
        integration_engine = PostMigrationIntegrationEngine(db_session)
        
        # 1. Configure cost tracking
        cost_tracking = integration_engine.configure_cost_tracking(project_id)
        
        assert 'cost_centers_created' in cost_tracking
        assert len(cost_tracking['cost_centers_created']) == 2
        assert 'attribution_rules' in cost_tracking
        assert len(cost_tracking['attribution_rules']) > 0
        
        # 2. Transfer organizational structure to FinOps
        finops_transfer = integration_engine.transfer_organizational_structure(
            project_id=project_id
        )
        
        assert 'teams_created' in finops_transfer
        assert 'projects_created' in finops_transfer
        assert 'cost_centers_created' in finops_transfer
        assert len(finops_transfer['teams_created']) == 2
        assert len(finops_transfer['projects_created']) == 2
        
        # 3. Enable FinOps features
        finops_features = integration_engine.enable_finops_features(
            project_id=project_id,
            provider='AWS'
        )
        
        assert 'features_enabled' in finops_features
        feature_names = [f['feature'] for f in finops_features['features_enabled']]
        
        # Verify key features are enabled
        assert 'waste_detection' in feature_names
        assert 'ri_optimization' in feature_names
        assert 'cost_anomaly_detection' in feature_names
        assert 'budget_management' in feature_names
        assert 'tagging_compliance' in feature_names
        
        # 4. Capture baselines
        sample_resources = [
            {'resource_id': 'i-123', 'resource_type': 'compute_instance'},
            {'resource_id': 'vol-456', 'resource_type': 'storage_volume'},
            {'resource_id': 'db-789', 'resource_type': 'database'}
        ]
        
        baseline = integration_engine.capture_baselines(
            project_id=project_id,
            resources=sample_resources
        )
        
        assert baseline['resource_count'] == 3
        assert baseline['total_monthly_cost'] > 0
        assert 'cost_by_service' in baseline
        
        # 5. Apply governance policies
        governance = integration_engine.apply_governance_policies(
            project_id=project_id,
            resources=sample_resources
        )
        
        assert 'resources_processed' in governance
        assert governance['resources_processed'] == 3
        
        # 6. Identify optimization opportunities
        optimization = integration_engine.identify_optimization_opportunities(
            project_id=project_id
        )
        
        assert 'opportunities' in optimization
        assert 'total_potential_savings' in optimization
        
        # 7. Generate final migration report
        report = integration_engine.generate_migration_report(project_id)
        
        assert report['project_id'] == project_id
        assert 'resources_migrated' in report
        assert 'success_rate' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
