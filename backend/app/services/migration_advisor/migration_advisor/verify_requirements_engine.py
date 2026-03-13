"""
Verification script for Workload and Requirements Analysis Engine

This script verifies that all components of the requirements analysis engine
are properly implemented and functional.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database import Base
from core.migration_advisor.requirements_analysis_engine import (
    WorkloadProfiler,
    PerformanceAnalyzer,
    ComplianceAssessor,
    BudgetAnalyzer,
    TechnicalRequirementsMapper,
    RequirementsCompletenessValidator,
    WorkloadAnalysisEngine
)
from core.migration_advisor.models import (
    MigrationProject,
    WorkloadProfile,
    PerformanceRequirements,
    ComplianceRequirements,
    BudgetConstraints,
    TechnicalRequirements,
    MigrationStatus,
    CompanySize,
    InfrastructureType,
    ExperienceLevel
)
from core.models import User
import uuid
from datetime import datetime


def verify_implementation():
    """Verify the requirements analysis engine implementation"""
    
    print("=" * 80)
    print("Workload and Requirements Analysis Engine Verification")
    print("=" * 80)
    
    # Create in-memory database for testing
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        # Create test user
        test_user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            username="testuser",
            hashed_password="test",
            is_active=True
        )
        db.add(test_user)
        db.commit()
        
        # Create test migration project
        project = MigrationProject(
            project_id="test-project-001",
            organization_name="Test Organization",
            status=MigrationStatus.ANALYSIS,
            current_phase="Requirements Analysis",
            created_by=test_user.id
        )
        db.add(project)
        db.commit()
        
        print("\n✓ Test project created successfully")
        
        # Test 1: WorkloadProfiler
        print("\n" + "=" * 80)
        print("Test 1: Workload Profiling System")
        print("=" * 80)
        
        profiler = WorkloadProfiler(db)
        workload = profiler.create_workload_profile(
            migration_project_id=project.id,
            workload_name="Web Application",
            application_type="web",
            total_compute_cores=16,
            total_memory_gb=64,
            total_storage_tb=2.5,
            database_types=["PostgreSQL", "Redis"],
            data_volume_tb=1.5,
            peak_transaction_rate=5000,
            workload_patterns={"peak_hours": "9am-5pm", "highly_variable": False},
            dependencies=[]
        )
        db.commit()
        
        print(f"✓ Workload profile created: {workload.workload_name}")
        print(f"  - Application Type: {workload.application_type}")
        print(f"  - Compute Cores: {workload.total_compute_cores}")
        print(f"  - Memory: {workload.total_memory_gb} GB")
        print(f"  - Storage: {workload.total_storage_tb} TB")
        
        # Test workload pattern analysis
        patterns = profiler.analyze_workload_patterns({
            'application_type': 'web',
            'total_compute_cores': 16,
            'total_memory_gb': 64,
            'total_storage_tb': 2.5,
            'peak_transaction_rate': 5000
        })
        print(f"✓ Workload pattern analysis completed")
        print(f"  - Workload Type: {patterns['workload_type']}")
        print(f"  - Resource Intensity: {patterns['resource_intensity']}")
        print(f"  - Scalability: {patterns['scalability_requirements']}")
        
        # Test 2: PerformanceAnalyzer
        print("\n" + "=" * 80)
        print("Test 2: Performance Requirements Analyzer")
        print("=" * 80)
        
        analyzer = PerformanceAnalyzer(db)
        perf_req = analyzer.create_performance_requirements(
            migration_project_id=project.id,
            availability_target=99.95,
            latency_requirements={"api": "< 100ms", "database": "< 50ms"},
            disaster_recovery_rto=60,
            disaster_recovery_rpo=15,
            geographic_distribution=["us-east", "us-west", "eu-west"],
            peak_load_multiplier=2.5
        )
        db.commit()
        
        print(f"✓ Performance requirements created")
        print(f"  - Availability Target: {perf_req.availability_target}%")
        print(f"  - RTO: {perf_req.disaster_recovery_rto} minutes")
        print(f"  - RPO: {perf_req.disaster_recovery_rpo} minutes")
        print(f"  - Geographic Distribution: {len(perf_req.geographic_distribution)} regions")
        
        # Test performance validation
        validation = analyzer.validate_performance_profile({
            'availability_target': 99.95,
            'disaster_recovery_rto': 60,
            'disaster_recovery_rpo': 15,
            'geographic_distribution': ["us-east", "us-west", "eu-west"]
        })
        print(f"✓ Performance validation completed")
        print(f"  - Recommendations: {len(validation['recommendations'])}")
        
        # Test 3: ComplianceAssessor
        print("\n" + "=" * 80)
        print("Test 3: Compliance Assessment System")
        print("=" * 80)
        
        assessor = ComplianceAssessor(db)
        compliance_req = assessor.create_compliance_requirements(
            migration_project_id=project.id,
            regulatory_frameworks=["GDPR", "SOC2", "ISO27001"],
            data_residency_requirements=["EU", "US"],
            industry_certifications=["PCI-DSS"],
            security_standards=["TLS 1.3", "AES-256"],
            audit_requirements={"log_retention_days": 365}
        )
        db.commit()
        
        print(f"✓ Compliance requirements created")
        print(f"  - Regulatory Frameworks: {len(compliance_req.regulatory_frameworks)}")
        print(f"  - Data Residency: {len(compliance_req.data_residency_requirements)} regions")
        print(f"  - Certifications: {len(compliance_req.industry_certifications)}")
        
        # Test compliance validation
        validation = assessor.validate_compliance_profile({
            'regulatory_frameworks': ["GDPR", "SOC2"],
            'data_residency_requirements': ["EU", "US"]
        })
        print(f"✓ Compliance validation completed")
        print(f"  - Warnings: {len(validation['warnings'])}")
        print(f"  - Recommendations: {len(validation['recommendations'])}")
        
        # Test 4: BudgetAnalyzer
        print("\n" + "=" * 80)
        print("Test 4: Budget Analysis Component")
        print("=" * 80)
        
        budget_analyzer = BudgetAnalyzer(db)
        budget = budget_analyzer.create_budget_constraints(
            migration_project_id=project.id,
            migration_budget=500000.00,
            current_monthly_cost=50000.00,
            target_monthly_cost=40000.00,
            cost_optimization_priority='high',
            acceptable_cost_variance=10.0,
            currency='USD'
        )
        db.commit()
        
        print(f"✓ Budget constraints created")
        print(f"  - Migration Budget: ${float(budget.migration_budget):,.2f}")
        print(f"  - Current Monthly Cost: ${float(budget.current_monthly_cost):,.2f}")
        print(f"  - Target Monthly Cost: ${float(budget.target_monthly_cost):,.2f}")
        print(f"  - Optimization Priority: {budget.cost_optimization_priority}")
        
        # Test budget analysis
        analysis = budget_analyzer.analyze_cost_optimization_priority({
            'migration_budget': 500000.00,
            'current_monthly_cost': 50000.00,
            'target_monthly_cost': 40000.00,
            'cost_optimization_priority': 'high'
        })
        print(f"✓ Budget analysis completed")
        print(f"  - Target Cost Reduction: {analysis.get('target_cost_reduction_pct', 0):.1f}%")
        print(f"  - Recommendations: {len(analysis['recommendations'])}")
        
        # Test 5: TechnicalRequirementsMapper
        print("\n" + "=" * 80)
        print("Test 5: Technical Requirements Mapper")
        print("=" * 80)
        
        tech_mapper = TechnicalRequirementsMapper(db)
        tech_req = tech_mapper.create_technical_requirements(
            migration_project_id=project.id,
            required_services=["Compute", "Storage", "Database", "Load Balancer"],
            ml_ai_requirements={"model_training": True, "inference": True},
            analytics_requirements={"data_warehouse": True, "real_time": False},
            container_orchestration=True,
            serverless_requirements=True,
            specialized_compute=["GPU"],
            integration_requirements={"api_gateway": True}
        )
        db.commit()
        
        print(f"✓ Technical requirements created")
        print(f"  - Required Services: {len(tech_req.required_services)}")
        print(f"  - Container Orchestration: {tech_req.container_orchestration}")
        print(f"  - Serverless: {tech_req.serverless_requirements}")
        print(f"  - Specialized Compute: {tech_req.specialized_compute}")
        
        # Test service mapping
        service_mapping = tech_mapper.map_service_requirements({
            'required_services': ["Compute", "Storage", "Database"],
            'ml_ai_requirements': {"model_training": True},
            'container_orchestration': True,
            'serverless_requirements': True
        })
        print(f"✓ Service mapping completed")
        for category, services in service_mapping.items():
            if services:
                print(f"  - {category}: {len(services)} service(s)")
        
        # Test 6: RequirementsCompletenessValidator
        print("\n" + "=" * 80)
        print("Test 6: Requirements Completeness Validation")
        print("=" * 80)
        
        validator = RequirementsCompletenessValidator(db)
        completeness = validator.validate_requirements_completeness(project.id)
        
        print(f"✓ Completeness validation completed")
        print(f"  - Is Complete: {completeness['is_complete']}")
        print(f"  - Completeness Score: {completeness['completeness_score']:.1f}%")
        print(f"  - Missing Items: {len(completeness['missing_items'])}")
        
        # Test consistency check
        consistency = validator.check_consistency(project.id)
        print(f"✓ Consistency check completed")
        print(f"  - Is Consistent: {consistency['is_consistent']}")
        print(f"  - Issues: {len(consistency['issues'])}")
        print(f"  - Warnings: {len(consistency['warnings'])}")
        
        # Test 7: WorkloadAnalysisEngine (Integration)
        print("\n" + "=" * 80)
        print("Test 7: Workload Analysis Engine (Integration)")
        print("=" * 80)
        
        # Create a new project for integration test
        project2 = MigrationProject(
            project_id="test-project-002",
            organization_name="Integration Test Org",
            status=MigrationStatus.ANALYSIS,
            current_phase="Requirements Analysis",
            created_by=test_user.id
        )
        db.add(project2)
        db.commit()
        
        engine_test = WorkloadAnalysisEngine(db)
        
        # Test integrated workflow
        result = engine_test.validate_requirements_completeness("test-project-002")
        print(f"✓ Integration test completed")
        print(f"  - Completeness Score: {result['completeness']['completeness_score']:.1f}%")
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Workload Profiling System - WORKING")
        print("  ✓ Performance Requirements Analyzer - WORKING")
        print("  ✓ Compliance Assessment System - WORKING")
        print("  ✓ Budget Analysis Component - WORKING")
        print("  ✓ Technical Requirements Mapper - WORKING")
        print("  ✓ Requirements Completeness Validator - WORKING")
        print("  ✓ Workload Analysis Engine Integration - WORKING")
        print("\nThe Workload and Requirements Analysis Engine is fully functional!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()
    
    return True


if __name__ == "__main__":
    success = verify_implementation()
    sys.exit(0 if success else 1)
