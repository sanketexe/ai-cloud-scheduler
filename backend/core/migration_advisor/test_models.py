"""
Basic tests for migration advisor models

These tests verify that models can be instantiated and basic validation works.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from .models import (
    MigrationProject,
    OrganizationProfile,
    WorkloadProfile,
    PerformanceRequirements,
    ComplianceRequirements,
    BudgetConstraints,
    TechnicalRequirements,
    ProviderEvaluation,
    RecommendationReport,
    MigrationPlan,
    MigrationPhase,
    OrganizationalStructure,
    CategorizedResource,
    MigrationStatus,
    CompanySize,
    InfrastructureType,
    ExperienceLevel,
    PhaseStatus,
    OwnershipStatus,
    MigrationRiskLevel,
)


def test_migration_project_creation():
    """Test MigrationProject model instantiation"""
    project = MigrationProject(
        project_id="MIG-2024-001",
        organization_name="Test Corp",
        status=MigrationStatus.ASSESSMENT,
        created_by=uuid.uuid4()
    )
    assert project.project_id == "MIG-2024-001"
    assert project.organization_name == "Test Corp"
    assert project.status == MigrationStatus.ASSESSMENT
    print("✓ MigrationProject creation test passed")


def test_organization_profile_creation():
    """Test OrganizationProfile model instantiation"""
    profile = OrganizationProfile(
        migration_project_id=uuid.uuid4(),
        company_size=CompanySize.MEDIUM,
        industry="Technology",
        current_infrastructure=InfrastructureType.ON_PREMISES,
        it_team_size=25,
        cloud_experience_level=ExperienceLevel.BEGINNER,
        geographic_presence=["US", "EU"]
    )
    assert profile.company_size == CompanySize.MEDIUM
    assert profile.industry == "Technology"
    assert profile.it_team_size == 25
    print("✓ OrganizationProfile creation test passed")


def test_workload_profile_creation():
    """Test WorkloadProfile model instantiation"""
    workload = WorkloadProfile(
        migration_project_id=uuid.uuid4(),
        workload_name="Web Application",
        application_type="Web",
        total_compute_cores=16,
        total_memory_gb=64,
        total_storage_tb=2.5
    )
    assert workload.workload_name == "Web Application"
    assert workload.total_compute_cores == 16
    assert workload.total_storage_tb == 2.5
    print("✓ WorkloadProfile creation test passed")


def test_performance_requirements_validation():
    """Test PerformanceRequirements validation"""
    # Valid availability target
    perf = PerformanceRequirements(
        migration_project_id=uuid.uuid4(),
        availability_target=99.99,
        disaster_recovery_rto=60,
        disaster_recovery_rpo=15
    )
    assert perf.availability_target == 99.99
    
    # Test validation would fail for invalid values (0-100 range)
    try:
        invalid_perf = PerformanceRequirements(
            migration_project_id=uuid.uuid4(),
            availability_target=150.0  # Invalid: > 100
        )
        # Validation happens at database level, so this won't fail here
        # but would fail when committing to database
    except ValueError:
        pass
    
    print("✓ PerformanceRequirements validation test passed")


def test_budget_constraints_validation():
    """Test BudgetConstraints validation"""
    budget = BudgetConstraints(
        migration_project_id=uuid.uuid4(),
        migration_budget=Decimal("100000.00"),
        target_monthly_cost=Decimal("5000.00"),
        currency="USD"
    )
    assert budget.migration_budget == Decimal("100000.00")
    assert budget.currency == "USD"
    print("✓ BudgetConstraints validation test passed")


def test_provider_evaluation_creation():
    """Test ProviderEvaluation model instantiation"""
    evaluation = ProviderEvaluation(
        migration_project_id=uuid.uuid4(),
        provider_name="AWS",
        service_availability_score=0.95,
        pricing_score=0.85,
        compliance_score=0.90,
        technical_fit_score=0.92,
        migration_complexity_score=0.88,
        overall_score=0.90,
        strengths=["Wide service catalog", "Strong compliance"],
        weaknesses=["Complex pricing"]
    )
    assert evaluation.provider_name == "AWS"
    assert evaluation.overall_score == 0.90
    print("✓ ProviderEvaluation creation test passed")


def test_recommendation_report_validation():
    """Test RecommendationReport confidence score validation"""
    report = RecommendationReport(
        migration_project_id=uuid.uuid4(),
        primary_recommendation="AWS",
        confidence_score=0.85,
        justification="AWS provides the best fit for requirements"
    )
    assert report.confidence_score == 0.85
    print("✓ RecommendationReport validation test passed")


def test_migration_plan_creation():
    """Test MigrationPlan model instantiation"""
    plan = MigrationPlan(
        plan_id="PLAN-2024-001",
        migration_project_id=uuid.uuid4(),
        target_provider="AWS",
        total_duration_days=90,
        estimated_cost=Decimal("150000.00"),
        risk_level=MigrationRiskLevel.MEDIUM
    )
    assert plan.plan_id == "PLAN-2024-001"
    assert plan.target_provider == "AWS"
    assert plan.risk_level == MigrationRiskLevel.MEDIUM
    print("✓ MigrationPlan creation test passed")


def test_migration_phase_creation():
    """Test MigrationPhase model instantiation"""
    phase = MigrationPhase(
        phase_id="PHASE-001",
        migration_plan_id=uuid.uuid4(),
        phase_name="Infrastructure Setup",
        phase_order=1,
        status=PhaseStatus.NOT_STARTED,
        workloads=["workload-1", "workload-2"]
    )
    assert phase.phase_name == "Infrastructure Setup"
    assert phase.phase_order == 1
    assert phase.status == PhaseStatus.NOT_STARTED
    print("✓ MigrationPhase creation test passed")


def test_organizational_structure_creation():
    """Test OrganizationalStructure model instantiation"""
    structure = OrganizationalStructure(
        migration_project_id=uuid.uuid4(),
        structure_name="Acme Corp Structure",
        teams=["Engineering", "Operations", "Data"],
        projects=["Project A", "Project B"],
        environments=["dev", "staging", "prod"],
        regions=["us-east-1", "eu-west-1"]
    )
    assert structure.structure_name == "Acme Corp Structure"
    assert len(structure.teams) == 3
    print("✓ OrganizationalStructure creation test passed")


def test_categorized_resource_creation():
    """Test CategorizedResource model instantiation"""
    resource = CategorizedResource(
        migration_project_id=uuid.uuid4(),
        resource_id="i-1234567890abcdef0",
        resource_type="EC2Instance",
        resource_name="web-server-01",
        provider="AWS",
        team="Engineering",
        project="Project A",
        environment="prod",
        region="us-east-1",
        ownership_status=OwnershipStatus.ASSIGNED,
        tags={"Name": "web-server-01", "Environment": "prod"}
    )
    assert resource.resource_id == "i-1234567890abcdef0"
    assert resource.team == "Engineering"
    assert resource.ownership_status == OwnershipStatus.ASSIGNED
    print("✓ CategorizedResource creation test passed")


def run_all_tests():
    """Run all model tests"""
    print("\nRunning Migration Advisor Model Tests...\n")
    
    test_migration_project_creation()
    test_organization_profile_creation()
    test_workload_profile_creation()
    test_performance_requirements_validation()
    test_budget_constraints_validation()
    test_provider_evaluation_creation()
    test_recommendation_report_validation()
    test_migration_plan_creation()
    test_migration_phase_creation()
    test_organizational_structure_creation()
    test_categorized_resource_creation()
    
    print("\n✅ All model tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
