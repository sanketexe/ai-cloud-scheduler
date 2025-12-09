"""
Performance Test Suite for Cloud Migration Advisor

This module contains performance tests for the migration advisor system,
including load tests for API endpoints, recommendation engine performance
with large datasets, and resource discovery/organization scalability.

Tests cover:
- API endpoint load testing
- Recommendation engine performance with large datasets
- Resource discovery and organization scalability
- Database query performance
"""

import pytest
import time
from datetime import datetime
from decimal import Decimal
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..database import Base
from ..models import User, UserRole
from .models import (
    MigrationProject, OrganizationProfile, WorkloadProfile,
    PerformanceRequirements, ComplianceRequirements, BudgetConstraints,
    CategorizedResource, MigrationStatus, CompanySize,
    InfrastructureType, ExperienceLevel, OwnershipStatus
)
from .assessment_engine import MigrationAssessmentEngine
from .requirements_analysis_engine import RequirementsAnalysisEngine
from .recommendation_engine import CloudProviderRecommendationEngine
from .resource_organization_engine import ResourceOrganizationEngine


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
        email="perf_test@example.com",
        password_hash="hashed_password",
        first_name="Performance",
        last_name="Test",
        role=UserRole.ADMIN
    )
    db_session.add(user)
    db_session.commit()
    return user


# Performance Test Utilities

class PerformanceMetrics:
    """Utility class to track performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.operations_count = 0
        self.errors_count = 0
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing and calculate duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def record_operation(self, success=True):
        """Record an operation"""
        self.operations_count += 1
        if not success:
            self.errors_count += 1
    
    def get_ops_per_second(self):
        """Calculate operations per second"""
        if self.duration and self.duration > 0:
            return self.operations_count / self.duration
        return 0
    
    def get_summary(self):
        """Get performance summary"""
        return {
            'total_operations': self.operations_count,
            'total_errors': self.errors_count,
            'duration_seconds': self.duration,
            'ops_per_second': self.get_ops_per_second(),
            'success_rate': (self.operations_count - self.errors_count) / self.operations_count if self.operations_count > 0 else 0
        }


# Performance Tests

class TestAPIEndpointPerformance:
    """Test API endpoint performance under load"""
    
    def test_project_creation_load(self, db_session, test_user):
        """Test project creation endpoint under load"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        metrics = PerformanceMetrics()
        
        num_projects = 100
        
        metrics.start()
        
        for i in range(num_projects):
            try:
                project_result = assessment_engine.initiate_migration_assessment(
                    organization_name=f"Load Test Org {i}",
                    created_by_user_id=test_user.id
                )
                metrics.record_operation(success=True)
            except Exception as e:
                metrics.record_operation(success=False)
        
        metrics.stop()
        
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary['total_operations'] == num_projects
        assert summary['success_rate'] >= 0.95  # 95% success rate
        assert summary['duration_seconds'] < 30  # Should complete in under 30 seconds
        assert summary['ops_per_second'] > 3  # At least 3 projects per second
        
        print(f"\nProject Creation Performance:")
        print(f"  Total: {summary['total_operations']}")
        print(f"  Duration: {summary['duration_seconds']:.2f}s")
        print(f"  Throughput: {summary['ops_per_second']:.2f} ops/sec")
    
    def test_concurrent_project_access(self, db_session, test_user):
        """Test concurrent access to projects"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        # Create test projects
        project_ids = []
        for i in range(10):
            result = assessment_engine.initiate_migration_assessment(
                organization_name=f"Concurrent Test {i}",
                created_by_user_id=test_user.id
            )
            project_ids.append(result['project_id'])
        
        db_session.commit()
        
        metrics = PerformanceMetrics()
        
        def access_project(project_id):
            """Access a project"""
            try:
                project = db_session.query(MigrationProject).filter(
                    MigrationProject.project_id == project_id
                ).first()
                return project is not None
            except Exception:
                return False
        
        metrics.start()
        
        # Simulate concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(100):  # 100 concurrent accesses
                project_id = project_ids[_ % len(project_ids)]
                future = executor.submit(access_project, project_id)
                futures.append(future)
            
            for future in as_completed(futures):
                success = future.result()
                metrics.record_operation(success=success)
        
        metrics.stop()
        
        summary = metrics.get_summary()
        
        assert summary['success_rate'] >= 0.95
        assert summary['duration_seconds'] < 10
        
        print(f"\nConcurrent Access Performance:")
        print(f"  Total: {summary['total_operations']}")
        print(f"  Duration: {summary['duration_seconds']:.2f}s")
        print(f"  Throughput: {summary['ops_per_second']:.2f} ops/sec")


class TestRecommendationEnginePerformance:
    """Test recommendation engine performance with large datasets"""
    
    def test_recommendation_with_many_workloads(self, db_session, test_user):
        """Test recommendation generation with many workloads"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        requirements_engine = RequirementsAnalysisEngine(db_session)
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Large Workload Test",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Add organization profile
        org_profile = {
            'company_size': 'enterprise',
            'industry': 'Technology',
            'current_infrastructure': 'hybrid',
            'it_team_size': 100,
            'cloud_experience_level': 'advanced',
            'geographic_presence': ['Global']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=org_profile
        )
        
        # Add many workloads
        num_workloads = 50
        metrics = PerformanceMetrics()
        
        metrics.start()
        
        for i in range(num_workloads):
            workload_data = {
                'workload_name': f'Workload {i}',
                'application_type': ['web', 'database', 'application', 'analytics'][i % 4],
                'total_compute_cores': 8 + (i % 10),
                'total_memory_gb': 32 + (i % 20) * 4,
                'total_storage_tb': 1.0 + (i % 5),
                'data_volume_tb': 0.5 + (i % 3)
            }
            
            try:
                requirements_engine.analyze_workloads(
                    project_id=project_id,
                    workload_data=workload_data
                )
                metrics.record_operation(success=True)
            except Exception:
                metrics.record_operation(success=False)
        
        metrics.stop()
        
        workload_summary = metrics.get_summary()
        
        # Now test recommendation generation
        rec_metrics = PerformanceMetrics()
        rec_metrics.start()
        
        try:
            recommendations = recommendation_engine.generate_recommendations(project_id)
            rec_metrics.record_operation(success=True)
        except Exception:
            rec_metrics.record_operation(success=False)
        
        rec_metrics.stop()
        
        rec_summary = rec_metrics.get_summary()
        
        # Performance assertions
        assert workload_summary['success_rate'] >= 0.95
        assert rec_summary['success_rate'] == 1.0
        assert rec_summary['duration_seconds'] < 10  # Should complete in under 10 seconds
        
        print(f"\nRecommendation Engine Performance (50 workloads):")
        print(f"  Workload Creation: {workload_summary['duration_seconds']:.2f}s")
        print(f"  Recommendation Generation: {rec_summary['duration_seconds']:.2f}s")
    
    def test_recommendation_with_complex_requirements(self, db_session, test_user):
        """Test recommendation with complex requirements"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        requirements_engine = RequirementsAnalysisEngine(db_session)
        recommendation_engine = CloudProviderRecommendationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Complex Requirements Test",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Add profile
        org_profile = {
            'company_size': 'large',
            'industry': 'Healthcare',
            'current_infrastructure': 'hybrid',
            'it_team_size': 75,
            'cloud_experience_level': 'intermediate',
            'geographic_presence': ['North America', 'Europe', 'Asia']
        }
        
        assessment_engine.collect_organization_profile(
            project_id=project_id,
            profile_data=org_profile
        )
        
        # Add workloads
        for i in range(10):
            workload_data = {
                'workload_name': f'Healthcare Workload {i}',
                'application_type': 'application',
                'total_compute_cores': 16,
                'total_memory_gb': 64,
                'total_storage_tb': 5.0,
                'data_volume_tb': 3.0
            }
            requirements_engine.analyze_workloads(
                project_id=project_id,
                workload_data=workload_data
            )
        
        # Add complex compliance requirements
        compliance_data = {
            'regulatory_frameworks': ['HIPAA', 'HITRUST', 'SOC2', 'GDPR', 'ISO27001'],
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
        
        # Add strict performance requirements
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
        
        # Test recommendation generation
        metrics = PerformanceMetrics()
        metrics.start()
        
        try:
            recommendations = recommendation_engine.generate_recommendations(project_id)
            metrics.record_operation(success=True)
            
            # Verify recommendations are comprehensive
            assert 'ranked_providers' in recommendations
            assert len(recommendations['ranked_providers']) > 0
        except Exception:
            metrics.record_operation(success=False)
        
        metrics.stop()
        
        summary = metrics.get_summary()
        
        assert summary['success_rate'] == 1.0
        assert summary['duration_seconds'] < 15  # Complex requirements may take longer
        
        print(f"\nComplex Requirements Recommendation Performance:")
        print(f"  Duration: {summary['duration_seconds']:.2f}s")


class TestResourceOrganizationPerformance:
    """Test resource discovery and organization scalability"""
    
    def test_large_scale_resource_categorization(self, db_session, test_user):
        """Test categorization of large number of resources"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        org_engine = ResourceOrganizationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Large Scale Resource Test",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Define organizational structure
        org_structure_data = {
            'structure_name': 'Large Scale Structure',
            'teams': [
                {'name': f'Team {i}', 'owner': f'team{i}@example.com'}
                for i in range(10)
            ],
            'projects': [
                {'name': f'Project {i}', 'team': f'Team {i % 10}'}
                for i in range(20)
            ],
            'environments': ['dev', 'staging', 'prod'],
            'regions': ['us-east-1', 'us-west-2', 'eu-west-1']
        }
        
        org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        # Generate large number of resources
        num_resources = 1000
        resources = []
        
        for i in range(num_resources):
            resource = {
                'resource_id': f'resource-{i}',
                'resource_type': ['EC2Instance', 'EBSVolume', 'RDSInstance', 'S3Bucket'][i % 4],
                'resource_name': f'resource-{i}',
                'provider': 'AWS',
                'region': ['us-east-1', 'us-west-2', 'eu-west-1'][i % 3],
                'tags': {
                    'Name': f'resource-{i}',
                    'Environment': ['dev', 'staging', 'prod'][i % 3]
                }
            }
            resources.append(resource)
        
        # Test categorization performance
        metrics = PerformanceMetrics()
        metrics.start()
        
        try:
            result = org_engine.categorize_resources(
                project_id=project_id,
                resources=resources
            )
            metrics.record_operation(success=True)
            
            assert 'categorized_count' in result
            assert result['categorized_count'] > 0
        except Exception as e:
            print(f"Error: {e}")
            metrics.record_operation(success=False)
        
        metrics.stop()
        
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary['success_rate'] == 1.0
        assert summary['duration_seconds'] < 30  # Should handle 1000 resources in under 30 seconds
        
        resources_per_second = num_resources / summary['duration_seconds']
        assert resources_per_second > 30  # At least 30 resources per second
        
        print(f"\nResource Categorization Performance (1000 resources):")
        print(f"  Duration: {summary['duration_seconds']:.2f}s")
        print(f"  Throughput: {resources_per_second:.2f} resources/sec")
    
    def test_hierarchical_view_generation_performance(self, db_session, test_user):
        """Test performance of hierarchical view generation"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        org_engine = ResourceOrganizationEngine(db_session)
        
        # Create project
        project_result = assessment_engine.initiate_migration_assessment(
            organization_name="Hierarchy Performance Test",
            created_by_user_id=test_user.id
        )
        project_id = project_result['project_id']
        
        # Define organizational structure
        org_structure_data = {
            'structure_name': 'Hierarchy Test Structure',
            'teams': [{'name': f'Team {i}', 'owner': f'team{i}@example.com'} for i in range(5)],
            'projects': [{'name': f'Project {i}', 'team': f'Team {i % 5}'} for i in range(10)],
            'environments': ['dev', 'staging', 'prod'],
            'regions': ['us-east-1', 'us-west-2']
        }
        
        org_engine.define_organizational_structure(
            project_id=project_id,
            structure_data=org_structure_data
        )
        
        # Create categorized resources
        num_resources = 500
        
        for i in range(num_resources):
            resource = CategorizedResource(
                migration_project_id=db_session.query(MigrationProject).filter(
                    MigrationProject.project_id == project_id
                ).first().id,
                resource_id=f'resource-{i}',
                resource_type='EC2Instance',
                resource_name=f'instance-{i}',
                provider='AWS',
                team=f'Team {i % 5}',
                project=f'Project {i % 10}',
                environment=['dev', 'staging', 'prod'][i % 3],
                region=['us-east-1', 'us-west-2'][i % 2],
                ownership_status=OwnershipStatus.ASSIGNED,
                tags={'Name': f'instance-{i}'}
            )
            db_session.add(resource)
        
        db_session.commit()
        
        # Test hierarchy view generation
        metrics = PerformanceMetrics()
        metrics.start()
        
        try:
            hierarchy = org_engine.build_hierarchy_view(
                project_id=project_id,
                dimension='team'
            )
            metrics.record_operation(success=True)
            
            assert 'hierarchy' in hierarchy or 'nodes' in hierarchy
        except Exception as e:
            print(f"Error: {e}")
            metrics.record_operation(success=False)
        
        metrics.stop()
        
        summary = metrics.get_summary()
        
        assert summary['success_rate'] == 1.0
        assert summary['duration_seconds'] < 10
        
        print(f"\nHierarchy View Generation Performance (500 resources):")
        print(f"  Duration: {summary['duration_seconds']:.2f}s")


class TestDatabaseQueryPerformance:
    """Test database query performance"""
    
    def test_project_listing_performance(self, db_session, test_user):
        """Test performance of listing many projects"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        # Create many projects
        num_projects = 200
        
        for i in range(num_projects):
            assessment_engine.initiate_migration_assessment(
                organization_name=f"Query Test Org {i}",
                created_by_user_id=test_user.id
            )
        
        db_session.commit()
        
        # Test listing performance
        metrics = PerformanceMetrics()
        metrics.start()
        
        try:
            projects = db_session.query(MigrationProject).all()
            metrics.record_operation(success=True)
            
            assert len(projects) == num_projects
        except Exception:
            metrics.record_operation(success=False)
        
        metrics.stop()
        
        summary = metrics.get_summary()
        
        assert summary['success_rate'] == 1.0
        assert summary['duration_seconds'] < 2  # Should be very fast
        
        print(f"\nProject Listing Performance (200 projects):")
        print(f"  Duration: {summary['duration_seconds']:.2f}s")
    
    def test_filtered_query_performance(self, db_session, test_user):
        """Test performance of filtered queries"""
        assessment_engine = MigrationAssessmentEngine(db_session)
        
        # Create projects with different statuses
        for i in range(100):
            project_result = assessment_engine.initiate_migration_assessment(
                organization_name=f"Filter Test Org {i}",
                created_by_user_id=test_user.id
            )
            
            # Update some to different statuses
            if i % 3 == 0:
                project = db_session.query(MigrationProject).filter(
                    MigrationProject.project_id == project_result['project_id']
                ).first()
                project.status = MigrationStatus.ANALYSIS
        
        db_session.commit()
        
        # Test filtered query
        metrics = PerformanceMetrics()
        metrics.start()
        
        try:
            assessment_projects = db_session.query(MigrationProject).filter(
                MigrationProject.status == MigrationStatus.ASSESSMENT
            ).all()
            metrics.record_operation(success=True)
            
            assert len(assessment_projects) > 0
        except Exception:
            metrics.record_operation(success=False)
        
        metrics.stop()
        
        summary = metrics.get_summary()
        
        assert summary['success_rate'] == 1.0
        assert summary['duration_seconds'] < 1
        
        print(f"\nFiltered Query Performance:")
        print(f"  Duration: {summary['duration_seconds']:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
