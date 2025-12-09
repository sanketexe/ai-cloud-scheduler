# Cloud Migration Advisor Module

## Overview

The Cloud Migration Advisor module provides intelligent guidance for organizations migrating to the cloud. It includes assessment workflows, ML-based provider recommendations, migration planning, and automated resource organization.

## Directory Structure

```
migration_advisor/
├── __init__.py           # Module exports
├── models.py             # SQLAlchemy database models
├── README.md             # This file
└── (future components)
    ├── engines/          # Business logic engines
    ├── services/         # Service layer
    └── api/              # API endpoints
```

## Database Models

### Core Migration Models

- **MigrationProject**: Main project entity tracking migration lifecycle
- **OrganizationProfile**: Organization information and context
- **WorkloadProfile**: Application workload details
- **PerformanceRequirements**: Performance and availability needs
- **ComplianceRequirements**: Regulatory and compliance requirements
- **BudgetConstraints**: Budget and cost constraints
- **TechnicalRequirements**: Required cloud services and capabilities

### Recommendation Models

- **ProviderEvaluation**: Evaluation scores for each cloud provider
- **RecommendationReport**: Final recommendation with justification

### Migration Planning Models

- **MigrationPlan**: Comprehensive migration plan
- **MigrationPhase**: Individual migration phases with tracking

### Resource Organization Models

- **OrganizationalStructure**: Organizational hierarchy definition
- **CategorizedResource**: Resources with organizational categorization

## Enums

- **MigrationStatus**: Project status (assessment, analysis, recommendation, planning, execution, complete, cancelled)
- **CompanySize**: Organization size (small, medium, large, enterprise)
- **InfrastructureType**: Current infrastructure (on_premises, cloud, hybrid, multi_cloud)
- **ExperienceLevel**: Cloud experience (none, beginner, intermediate, advanced)
- **PhaseStatus**: Migration phase status (not_started, in_progress, completed, failed, rolled_back)
- **OwnershipStatus**: Resource ownership (assigned, unassigned, pending)
- **MigrationRiskLevel**: Risk levels (low, medium, high, critical)

## Database Migration

The database schema is managed through Alembic migrations:

```bash
# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

Migration file: `backend/alembic/versions/002_add_migration_advisor_tables.py`

## Usage

```python
from backend.core.migration_advisor import (
    MigrationProject,
    OrganizationProfile,
    WorkloadProfile,
    # ... other models
)

# Create a new migration project
project = MigrationProject(
    project_id="MIG-2024-001",
    organization_name="Acme Corp",
    status=MigrationStatus.ASSESSMENT,
    created_by=user_id
)
```

## Requirements Mapping

This implementation addresses the following requirements from the specification:

- **Requirement 1.1**: Migration project creation and tracking
- **Requirement 1.4**: Migration project lifecycle management
- **Requirements 1.2, 1.3**: Organization profiling and infrastructure analysis
- **Requirements 2.1-2.6**: Comprehensive requirements gathering
- **Requirements 3.1-3.6**: Provider evaluation and recommendations
- **Requirements 4.1-4.6**: Migration planning and execution
- **Requirements 5.1-5.6**: Resource organization and categorization
- **Requirements 6.1-6.6**: Multi-dimensional resource management

## Next Steps

Future components to be implemented:

1. Assessment engines for requirements gathering
2. ML-based recommendation engine
3. Migration planning engine
4. Resource discovery and organization engine
5. API endpoints for all functionality
6. UI components for migration wizard
