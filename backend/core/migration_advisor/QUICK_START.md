# Quick Start Guide: Migration Advisor Models

## Installation

1. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

2. Run database migrations:
```bash
cd backend
alembic upgrade head
```

## Basic Usage Examples

### Creating a Migration Project

```python
from backend.core.migration_advisor import (
    MigrationProject,
    MigrationStatus
)
from backend.core.database import get_db_session

# Create a new migration project
with get_db_session() as session:
    project = MigrationProject(
        project_id="MIG-2024-001",
        organization_name="Acme Corporation",
        status=MigrationStatus.ASSESSMENT,
        created_by=user_id
    )
    session.add(project)
    session.commit()
```

### Adding Organization Profile

```python
from backend.core.migration_advisor import (
    OrganizationProfile,
    CompanySize,
    InfrastructureType,
    ExperienceLevel
)

with get_db_session() as session:
    profile = OrganizationProfile(
        migration_project_id=project.id,
        company_size=CompanySize.MEDIUM,
        industry="Technology",
        current_infrastructure=InfrastructureType.ON_PREMISES,
        geographic_presence=["US", "EU", "APAC"],
        it_team_size=50,
        cloud_experience_level=ExperienceLevel.BEGINNER,
        additional_context={
            "current_providers": ["on-premises datacenter"],
            "pain_points": ["scaling", "maintenance costs"]
        }
    )
    session.add(profile)
    session.commit()
```

### Adding Workload Profiles

```python
from backend.core.migration_advisor import WorkloadProfile

with get_db_session() as session:
    workload = WorkloadProfile(
        migration_project_id=project.id,
        workload_name="E-commerce Platform",
        application_type="Web Application",
        total_compute_cores=64,
        total_memory_gb=256,
        total_storage_tb=5.0,
        database_types=["PostgreSQL", "Redis"],
        data_volume_tb=2.5,
        peak_transaction_rate=10000,
        workload_patterns={
            "peak_hours": "9am-9pm EST",
            "seasonal": "High during holidays"
        },
        dependencies=["payment-gateway", "inventory-system"]
    )
    session.add(workload)
    session.commit()
```

### Setting Requirements

```python
from backend.core.migration_advisor import (
    PerformanceRequirements,
    ComplianceRequirements,
    BudgetConstraints,
    TechnicalRequirements
)
from decimal import Decimal

with get_db_session() as session:
    # Performance requirements
    perf = PerformanceRequirements(
        migration_project_id=project.id,
        latency_requirements={
            "api_response": "< 200ms",
            "database_query": "< 50ms"
        },
        availability_target=99.95,
        disaster_recovery_rto=60,  # 1 hour
        disaster_recovery_rpo=15,  # 15 minutes
        geographic_distribution=["us-east", "us-west", "eu-west"],
        peak_load_multiplier=3.0
    )
    
    # Compliance requirements
    compliance = ComplianceRequirements(
        migration_project_id=project.id,
        regulatory_frameworks=["GDPR", "SOC2", "PCI-DSS"],
        data_residency_requirements=["EU", "US"],
        industry_certifications=["ISO 27001"],
        security_standards=["NIST", "CIS"]
    )
    
    # Budget constraints
    budget = BudgetConstraints(
        migration_project_id=project.id,
        current_monthly_cost=Decimal("50000.00"),
        migration_budget=Decimal("500000.00"),
        target_monthly_cost=Decimal("40000.00"),
        cost_optimization_priority="high",
        acceptable_cost_variance=0.15,  # 15%
        currency="USD"
    )
    
    # Technical requirements
    technical = TechnicalRequirements(
        migration_project_id=project.id,
        required_services=[
            "managed_database",
            "object_storage",
            "cdn",
            "load_balancer"
        ],
        ml_ai_requirements={
            "recommendation_engine": True,
            "fraud_detection": True
        },
        analytics_requirements={
            "real_time_analytics": True,
            "data_warehouse": True
        },
        container_orchestration=True,
        serverless_requirements=True,
        specialized_compute=["GPU for ML training"]
    )
    
    session.add_all([perf, compliance, budget, technical])
    session.commit()
```

### Recording Provider Evaluations

```python
from backend.core.migration_advisor import ProviderEvaluation

with get_db_session() as session:
    # AWS Evaluation
    aws_eval = ProviderEvaluation(
        migration_project_id=project.id,
        provider_name="AWS",
        service_availability_score=0.95,
        pricing_score=0.85,
        compliance_score=0.92,
        technical_fit_score=0.90,
        migration_complexity_score=0.88,
        overall_score=0.90,
        strengths=[
            "Comprehensive service catalog",
            "Strong compliance certifications",
            "Mature ML/AI services"
        ],
        weaknesses=[
            "Complex pricing model",
            "Steeper learning curve"
        ],
        detailed_analysis={
            "compute": "EC2, ECS, EKS available",
            "database": "RDS, DynamoDB, Aurora",
            "ml_services": "SageMaker, Rekognition"
        }
    )
    
    # GCP Evaluation
    gcp_eval = ProviderEvaluation(
        migration_project_id=project.id,
        provider_name="GCP",
        service_availability_score=0.88,
        pricing_score=0.90,
        compliance_score=0.85,
        technical_fit_score=0.87,
        migration_complexity_score=0.85,
        overall_score=0.87,
        strengths=[
            "Competitive pricing",
            "Strong data analytics",
            "Good Kubernetes support"
        ],
        weaknesses=[
            "Smaller service catalog",
            "Fewer compliance certifications"
        ]
    )
    
    session.add_all([aws_eval, gcp_eval])
    session.commit()
```

### Creating Recommendation Report

```python
from backend.core.migration_advisor import RecommendationReport

with get_db_session() as session:
    report = RecommendationReport(
        migration_project_id=project.id,
        primary_recommendation="AWS",
        confidence_score=0.85,
        key_differentiators=[
            "Best compliance fit for GDPR and PCI-DSS",
            "Most comprehensive ML/AI services",
            "Strong global presence"
        ],
        cost_comparison={
            "AWS": {"monthly": 42000, "migration": 480000},
            "GCP": {"monthly": 38000, "migration": 450000},
            "Azure": {"monthly": 45000, "migration": 520000}
        },
        risk_assessment={
            "technical_risk": "low",
            "cost_risk": "medium",
            "timeline_risk": "low"
        },
        justification="""
        AWS is recommended based on:
        1. Superior compliance certifications matching requirements
        2. Comprehensive ML/AI services for recommendation engine
        3. Strong track record with similar e-commerce migrations
        4. Global infrastructure supporting required regions
        """,
        scoring_weights={
            "service_availability": 0.25,
            "pricing": 0.20,
            "compliance": 0.30,
            "technical_fit": 0.15,
            "migration_complexity": 0.10
        },
        alternative_recommendations=["GCP", "Azure"]
    )
    session.add(report)
    session.commit()
```

### Creating Migration Plan

```python
from backend.core.migration_advisor import (
    MigrationPlan,
    MigrationPhase,
    MigrationRiskLevel,
    PhaseStatus
)

with get_db_session() as session:
    # Create migration plan
    plan = MigrationPlan(
        plan_id="PLAN-2024-001",
        migration_project_id=project.id,
        target_provider="AWS",
        total_duration_days=120,
        estimated_cost=Decimal("480000.00"),
        risk_level=MigrationRiskLevel.MEDIUM,
        dependencies_graph={
            "nodes": ["database", "app-servers", "cdn"],
            "edges": [["database", "app-servers"], ["app-servers", "cdn"]]
        },
        migration_waves=[
            {"wave": 1, "workloads": ["database"]},
            {"wave": 2, "workloads": ["app-servers"]},
            {"wave": 3, "workloads": ["cdn", "analytics"]}
        ],
        success_criteria=[
            "Zero data loss",
            "< 4 hours downtime",
            "All tests passing"
        ],
        rollback_strategy="Maintain parallel infrastructure for 30 days"
    )
    session.add(plan)
    session.flush()  # Get plan.id
    
    # Create migration phases
    phase1 = MigrationPhase(
        phase_id="PHASE-001",
        migration_plan_id=plan.id,
        phase_name="Infrastructure Setup",
        phase_order=1,
        status=PhaseStatus.NOT_STARTED,
        workloads=["network", "security"],
        prerequisites=["AWS account setup", "VPC configuration"],
        success_criteria=["Network connectivity verified", "Security groups configured"]
    )
    
    phase2 = MigrationPhase(
        phase_id="PHASE-002",
        migration_plan_id=plan.id,
        phase_name="Database Migration",
        phase_order=2,
        status=PhaseStatus.NOT_STARTED,
        workloads=["postgresql", "redis"],
        prerequisites=["Phase 1 complete", "Database backup verified"],
        success_criteria=["Data migrated", "Replication working"]
    )
    
    session.add_all([phase1, phase2])
    session.commit()
```

### Querying Migration Projects

```python
from sqlalchemy import select

with get_db_session() as session:
    # Get all active migration projects
    stmt = select(MigrationProject).where(
        MigrationProject.status.in_([
            MigrationStatus.ASSESSMENT,
            MigrationStatus.PLANNING,
            MigrationStatus.EXECUTION
        ])
    )
    active_projects = session.execute(stmt).scalars().all()
    
    # Get project with all relationships
    stmt = select(MigrationProject).where(
        MigrationProject.project_id == "MIG-2024-001"
    )
    project = session.execute(stmt).scalar_one()
    
    # Access relationships
    org_profile = project.organization_profile
    workloads = project.workload_profiles
    recommendation = project.recommendation_report
    plan = project.migration_plan
    phases = plan.phases if plan else []
```

## Common Patterns

### Updating Project Status

```python
with get_db_session() as session:
    project = session.get(MigrationProject, project_id)
    project.status = MigrationStatus.PLANNING
    project.current_phase = "Migration Planning"
    session.commit()
```

### Tracking Phase Progress

```python
with get_db_session() as session:
    phase = session.get(MigrationPhase, phase_id)
    phase.status = PhaseStatus.IN_PROGRESS
    phase.actual_start_date = datetime.now()
    session.commit()
```

### Filtering Resources by Team

```python
from backend.core.migration_advisor import CategorizedResource

with get_db_session() as session:
    stmt = select(CategorizedResource).where(
        CategorizedResource.team == "Engineering",
        CategorizedResource.environment == "prod"
    )
    resources = session.execute(stmt).scalars().all()
```

## Error Handling

```python
from sqlalchemy.exc import IntegrityError

try:
    with get_db_session() as session:
        project = MigrationProject(
            project_id="MIG-2024-001",  # Duplicate
            organization_name="Test",
            status=MigrationStatus.ASSESSMENT,
            created_by=user_id
        )
        session.add(project)
        session.commit()
except IntegrityError as e:
    print(f"Project ID already exists: {e}")
```

## Next Steps

After setting up the models:
1. Implement assessment engines (Task 2)
2. Build recommendation engine (Task 5)
3. Create API endpoints (Task 10)
4. Develop UI components (Task 11)
