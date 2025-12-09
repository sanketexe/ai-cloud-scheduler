"""
Tests for Migration Assessment Engine

This module tests the core functionality of the migration assessment engine
including project management, organization profiling, and timeline estimation.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..database import Base
from ..models import User, UserRole
from .models import (
    MigrationProject, OrganizationProfile, MigrationStatus,
    CompanySize, InfrastructureType, ExperienceLevel
)
from .assessment_engine import (
    MigrationProjectManager,
    OrganizationProfiler,
    AssessmentTimelineEstimator,
    MigrationAssessmentEngine
)


# Test fixtures

@pytest.fixture
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
        email="test@example.com",
        password_hash="hashed_password",
        first_name="Test",
        last_name="User",
        role=UserRole.ADMIN
    )
    db_session.add(user)
    db_session.commit()
    return user


# MigrationProjectManager Tests

def test_create_migration_project(db_session, test_user):
    """Test creating a new migration project"""
    manager = MigrationProjectManager(db_session)
    
    project = manager.create_migration_project(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    
    assert project is not None
    assert project.organization_name == "Test Corp"
    assert project.status == MigrationStatus.ASSESSMENT
    assert project.current_phase == "Initial Assessment"
    assert project.created_by == test_user.id
    assert project.project_id.startswith("mig-test-corp-")


def test_create_project_invalid_user(db_session):
    """Test creating project with invalid user ID"""
    manager = MigrationProjectManager(db_session)
    
    with pytest.raises(ValueError, match="User with id .* not found"):
        manager.create_migration_project(
            organization_name="Test Corp",
            created_by_user_id=uuid4()
        )


def test_create_project_empty_name(db_session, test_user):
    """Test creating project with empty organization name"""
    manager = MigrationProjectManager(db_session)
    
    with pytest.raises(ValueError, match="Organization name cannot be empty"):
        manager.create_migration_project(
            organization_name="",
            created_by_user_id=test_user.id
        )


def test_get_project(db_session, test_user):
    """Test retrieving a project by project_id"""
    manager = MigrationProjectManager(db_session)
    
    created_project = manager.create_migration_project(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    db_session.commit()
    
    retrieved_project = manager.get_project(created_project.project_id)
    
    assert retrieved_project is not None
    assert retrieved_project.id == created_project.id
    assert retrieved_project.organization_name == "Test Corp"


def test_update_project_status(db_session, test_user):
    """Test updating project status"""
    manager = MigrationProjectManager(db_session)
    
    project = manager.create_migration_project(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    db_session.commit()
    
    updated_project = manager.update_project_status(
        project_id=project.project_id,
        new_status=MigrationStatus.ANALYSIS,
        current_phase="Workload Analysis"
    )
    
    assert updated_project.status == MigrationStatus.ANALYSIS
    assert updated_project.current_phase == "Workload Analysis"


def test_invalid_status_transition(db_session, test_user):
    """Test invalid status transition"""
    manager = MigrationProjectManager(db_session)
    
    project = manager.create_migration_project(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    db_session.commit()
    
    with pytest.raises(ValueError, match="Invalid status transition"):
        manager.update_project_status(
            project_id=project.project_id,
            new_status=MigrationStatus.COMPLETE
        )


def test_list_projects(db_session, test_user):
    """Test listing projects"""
    manager = MigrationProjectManager(db_session)
    
    # Create multiple projects
    for i in range(3):
        manager.create_migration_project(
            organization_name=f"Test Corp {i}",
            created_by_user_id=test_user.id
        )
    db_session.commit()
    
    projects = manager.list_projects()
    
    assert len(projects) == 3


def test_list_projects_with_status_filter(db_session, test_user):
    """Test listing projects with status filter"""
    manager = MigrationProjectManager(db_session)
    
    # Create projects with different statuses
    project1 = manager.create_migration_project(
        organization_name="Test Corp 1",
        created_by_user_id=test_user.id
    )
    project2 = manager.create_migration_project(
        organization_name="Test Corp 2",
        created_by_user_id=test_user.id
    )
    db_session.commit()
    
    manager.update_project_status(
        project_id=project2.project_id,
        new_status=MigrationStatus.ANALYSIS
    )
    db_session.commit()
    
    assessment_projects = manager.list_projects(status=MigrationStatus.ASSESSMENT)
    analysis_projects = manager.list_projects(status=MigrationStatus.ANALYSIS)
    
    assert len(assessment_projects) == 1
    assert len(analysis_projects) == 1


# OrganizationProfiler Tests

def test_create_organization_profile(db_session, test_user):
    """Test creating an organization profile"""
    manager = MigrationProjectManager(db_session)
    profiler = OrganizationProfiler(db_session)
    
    project = manager.create_migration_project(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    db_session.commit()
    
    profile = profiler.create_organization_profile(
        migration_project_id=project.id,
        company_size=CompanySize.MEDIUM,
        industry="Technology",
        current_infrastructure=InfrastructureType.ON_PREMISES,
        it_team_size=15,
        cloud_experience_level=ExperienceLevel.BEGINNER,
        geographic_presence=["North America", "Europe"]
    )
    
    assert profile is not None
    assert profile.company_size == CompanySize.MEDIUM
    assert profile.industry == "Technology"
    assert profile.it_team_size == 15
    assert len(profile.geographic_presence) == 2


def test_create_profile_invalid_project(db_session):
    """Test creating profile with invalid project ID"""
    profiler = OrganizationProfiler(db_session)
    
    with pytest.raises(ValueError, match="Migration project .* not found"):
        profiler.create_organization_profile(
            migration_project_id=uuid4(),
            company_size=CompanySize.MEDIUM,
            industry="Technology",
            current_infrastructure=InfrastructureType.ON_PREMISES,
            it_team_size=15,
            cloud_experience_level=ExperienceLevel.BEGINNER
        )


def test_analyze_infrastructure_type(db_session):
    """Test infrastructure type analysis"""
    profiler = OrganizationProfiler(db_session)
    
    # Test on-premises
    result = profiler.analyze_infrastructure_type({
        'on_premises': True,
        'cloud_providers': []
    })
    assert result == InfrastructureType.ON_PREMISES
    
    # Test cloud
    result = profiler.analyze_infrastructure_type({
        'on_premises': False,
        'cloud_providers': ['AWS']
    })
    assert result == InfrastructureType.CLOUD
    
    # Test hybrid
    result = profiler.analyze_infrastructure_type({
        'on_premises': True,
        'cloud_providers': ['AWS']
    })
    assert result == InfrastructureType.HYBRID
    
    # Test multi-cloud
    result = profiler.analyze_infrastructure_type({
        'on_premises': False,
        'cloud_providers': ['AWS', 'GCP']
    })
    assert result == InfrastructureType.MULTI_CLOUD


# AssessmentTimelineEstimator Tests

def test_estimate_assessment_duration_small_company():
    """Test timeline estimation for small company"""
    estimator = AssessmentTimelineEstimator()
    
    result = estimator.estimate_assessment_duration(
        company_size=CompanySize.SMALL,
        current_infrastructure=InfrastructureType.ON_PREMISES,
        cloud_experience_level=ExperienceLevel.BEGINNER,
        it_team_size=3
    )
    
    assert 'estimated_days' in result
    assert 'estimated_completion_date' in result
    assert 'breakdown' in result
    assert result['estimated_days'] > 0


def test_estimate_assessment_duration_enterprise():
    """Test timeline estimation for enterprise company"""
    estimator = AssessmentTimelineEstimator()
    
    result = estimator.estimate_assessment_duration(
        company_size=CompanySize.ENTERPRISE,
        current_infrastructure=InfrastructureType.HYBRID,
        cloud_experience_level=ExperienceLevel.NONE,
        it_team_size=50
    )
    
    assert result['estimated_days'] > 0
    # Enterprise with hybrid infrastructure should take longer
    assert result['estimated_days'] >= 20


def test_estimate_with_experience_adjustment():
    """Test that experience level affects timeline"""
    estimator = AssessmentTimelineEstimator()
    
    # Beginner team
    beginner_result = estimator.estimate_assessment_duration(
        company_size=CompanySize.MEDIUM,
        current_infrastructure=InfrastructureType.ON_PREMISES,
        cloud_experience_level=ExperienceLevel.BEGINNER,
        it_team_size=10
    )
    
    # Advanced team
    advanced_result = estimator.estimate_assessment_duration(
        company_size=CompanySize.MEDIUM,
        current_infrastructure=InfrastructureType.ON_PREMISES,
        cloud_experience_level=ExperienceLevel.ADVANCED,
        it_team_size=10
    )
    
    # Advanced team should complete faster
    assert advanced_result['estimated_days'] < beginner_result['estimated_days']


# MigrationAssessmentEngine Integration Tests

def test_initiate_migration_assessment(db_session, test_user):
    """Test initiating a migration assessment"""
    engine = MigrationAssessmentEngine(db_session)
    
    result = engine.initiate_migration_assessment(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    
    assert 'project_id' in result
    assert 'project_uuid' in result
    assert result['organization_name'] == "Test Corp"
    assert result['status'] == "assessment"


def test_collect_organization_profile_integration(db_session, test_user):
    """Test collecting organization profile with timeline estimation"""
    engine = MigrationAssessmentEngine(db_session)
    
    # Create project
    project_result = engine.initiate_migration_assessment(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    db_session.commit()
    
    # Collect profile
    profile_data = {
        'company_size': 'medium',
        'industry': 'Technology',
        'current_infrastructure': 'on_premises',
        'it_team_size': 15,
        'cloud_experience_level': 'beginner',
        'geographic_presence': ['North America'],
        'additional_context': {}
    }
    
    result = engine.collect_organization_profile(
        project_id=project_result['project_id'],
        profile_data=profile_data
    )
    
    assert 'profile' in result
    assert 'timeline_estimation' in result
    assert result['profile']['company_size'] == 'medium'
    assert result['timeline_estimation']['estimated_days'] > 0


def test_validate_assessment_completeness(db_session, test_user):
    """Test assessment completeness validation"""
    engine = MigrationAssessmentEngine(db_session)
    
    # Create project without profile
    project_result = engine.initiate_migration_assessment(
        organization_name="Test Corp",
        created_by_user_id=test_user.id
    )
    db_session.commit()
    
    # Validate - should be incomplete
    validation = engine.validate_assessment_completeness(project_result['project_id'])
    
    assert validation['is_complete'] is False
    assert 'organization_profile' in validation['missing_items']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
