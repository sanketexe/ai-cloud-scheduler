# Requirements Analysis Engine - Quick Start Guide

## Overview
The Workload and Requirements Analysis Engine provides comprehensive analysis capabilities for cloud migration projects, covering workloads, performance, compliance, budget, and technical requirements.

## Usage Examples

### 1. Initialize the Engine

```python
from sqlalchemy.orm import Session
from core.migration_advisor import WorkloadAnalysisEngine

# Assuming you have a database session
engine = WorkloadAnalysisEngine(db_session)
```

### 2. Analyze Workloads

```python
# Create a workload profile
workload_result = engine.analyze_workloads(
    project_id="mig-acme-20241116-abc123",
    workload_data={
        "workload_name": "E-commerce Platform",
        "application_type": "web",
        "total_compute_cores": 32,
        "total_memory_gb": 128,
        "total_storage_tb": 5.0,
        "database_types": ["PostgreSQL", "Redis"],
        "data_volume_tb": 3.0,
        "peak_transaction_rate": 10000,
        "workload_patterns": {
            "peak_hours": "9am-9pm",
            "highly_variable": True
        }
    }
)
```

### 3. Assess Performance Requirements

```python
# Create performance requirements
perf_result = engine.assess_performance_requirements(
    project_id="mig-acme-20241116-abc123",
    perf_requirements={
        "availability_target": 99.99,
        "latency_requirements": {
            "api": "< 100ms",
            "database": "< 50ms"
        },
        "disaster_recovery_rto": 30,  # minutes
        "disaster_recovery_rpo": 5,   # minutes
        "geographic_distribution": ["us-east", "us-west", "eu-west"],
        "peak_load_multiplier": 3.0
    }
)
```

### 4. Evaluate Compliance Needs

```python
# Create compliance requirements
compliance_result = engine.evaluate_compliance_needs(
    project_id="mig-acme-20241116-abc123",
    compliance_data={
        "regulatory_frameworks": ["GDPR", "SOC2", "ISO27001"],
        "data_residency_requirements": ["EU", "US"],
        "industry_certifications": ["PCI-DSS"],
        "security_standards": ["TLS 1.3", "AES-256"],
        "audit_requirements": {
            "log_retention_days": 365,
            "audit_trail": True
        }
    }
)
```

### 5. Analyze Budget Constraints

```python
# Create budget constraints
budget_result = engine.analyze_budget_constraints(
    project_id="mig-acme-20241116-abc123",
    budget_data={
        "migration_budget": 500000.00,
        "current_monthly_cost": 50000.00,
        "target_monthly_cost": 40000.00,
        "cost_optimization_priority": "high",
        "acceptable_cost_variance": 10.0,
        "currency": "USD"
    }
)
```

### 6. Map Technical Requirements

```python
# Create technical requirements
tech_result = engine.map_technical_requirements(
    project_id="mig-acme-20241116-abc123",
    tech_requirements={
        "required_services": [
            "Compute", "Storage", "Database", 
            "Load Balancer", "CDN"
        ],
        "ml_ai_requirements": {
            "model_training": True,
            "inference": True,
            "gpu_required": True
        },
        "analytics_requirements": {
            "data_warehouse": True,
            "real_time_analytics": True
        },
        "container_orchestration": True,
        "serverless_requirements": True,
        "specialized_compute": ["GPU", "High-Memory"]
    }
)
```

### 7. Validate Requirements Completeness

```python
# Validate all requirements
validation_result = engine.validate_requirements_completeness(
    project_id="mig-acme-20241116-abc123"
)

print(f"Completeness: {validation_result['completeness']['completeness_score']}%")
print(f"Is Complete: {validation_result['completeness']['is_complete']}")
print(f"Missing Items: {validation_result['completeness']['missing_items']}")
print(f"Is Consistent: {validation_result['consistency']['is_consistent']}")
```

## API Endpoints

### Using the REST API

#### Create Workload Profile
```bash
POST /api/v1/migrations/{project_id}/workloads
Content-Type: application/json
Authorization: Bearer <token>

{
  "workload_name": "Web Application",
  "application_type": "web",
  "total_compute_cores": 16,
  "total_memory_gb": 64,
  "total_storage_tb": 2.5
}
```

#### Create Performance Requirements
```bash
POST /api/v1/migrations/{project_id}/performance-requirements
Content-Type: application/json
Authorization: Bearer <token>

{
  "availability_target": 99.95,
  "disaster_recovery_rto": 60,
  "disaster_recovery_rpo": 15,
  "geographic_distribution": ["us-east", "us-west"]
}
```

#### Create Compliance Requirements
```bash
POST /api/v1/migrations/{project_id}/compliance-requirements
Content-Type: application/json
Authorization: Bearer <token>

{
  "regulatory_frameworks": ["GDPR", "SOC2"],
  "data_residency_requirements": ["EU", "US"]
}
```

#### Create Budget Constraints
```bash
POST /api/v1/migrations/{project_id}/budget-constraints
Content-Type: application/json
Authorization: Bearer <token>

{
  "migration_budget": 500000.00,
  "current_monthly_cost": 50000.00,
  "target_monthly_cost": 40000.00,
  "cost_optimization_priority": "high"
}
```

#### Create Technical Requirements
```bash
POST /api/v1/migrations/{project_id}/technical-requirements
Content-Type: application/json
Authorization: Bearer <token>

{
  "required_services": ["Compute", "Storage", "Database"],
  "container_orchestration": true,
  "serverless_requirements": true
}
```

#### Validate Requirements
```bash
GET /api/v1/migrations/{project_id}/requirements/validation
Authorization: Bearer <token>
```

## Response Examples

### Workload Analysis Response
```json
{
  "workload_id": "uuid-here",
  "workload_name": "Web Application",
  "application_type": "web",
  "patterns": {
    "workload_type": "web-application",
    "resource_intensity": "medium",
    "scalability_requirements": "high-elasticity",
    "storage_characteristics": {
      "total_storage_tb": 2.5,
      "storage_type": "standard"
    }
  }
}
```

### Performance Assessment Response
```json
{
  "performance_requirements_id": "uuid-here",
  "availability_target": 99.95,
  "validation": {
    "is_valid": true,
    "warnings": [],
    "recommendations": [
      "High availability target requires multi-AZ deployment",
      "Low RTO requires active-active or hot standby configuration"
    ]
  }
}
```

### Validation Response
```json
{
  "completeness": {
    "is_complete": true,
    "missing_items": [],
    "warnings": [],
    "completeness_score": 100.0
  },
  "consistency": {
    "is_consistent": true,
    "issues": [],
    "warnings": []
  }
}
```

## Component Classes

### WorkloadProfiler
- Creates and manages workload profiles
- Analyzes workload patterns
- Classifies workload types

### PerformanceAnalyzer
- Manages performance requirements
- Validates performance profiles
- Provides performance recommendations

### ComplianceAssessor
- Manages compliance requirements
- Validates compliance profiles
- Identifies compliance issues

### BudgetAnalyzer
- Manages budget constraints
- Analyzes cost optimization priorities
- Calculates cost reduction targets

### TechnicalRequirementsMapper
- Manages technical requirements
- Maps services to categories
- Validates service mappings

### RequirementsCompletenessValidator
- Validates requirement completeness
- Checks consistency across categories
- Calculates completeness scores

## Best Practices

1. **Always validate completeness** before moving to the next phase
2. **Review recommendations** from each analyzer
3. **Check consistency** across requirement categories
4. **Update requirements** as project evolves
5. **Use validation warnings** to improve requirement quality

## Error Handling

All methods raise appropriate exceptions:
- `ValueError` - Invalid input or missing data
- `IntegrityError` - Database constraint violations
- `HTTPException` - API-level errors (in endpoints)

Always wrap calls in try-except blocks:

```python
try:
    result = engine.analyze_workloads(project_id, workload_data)
    db.commit()
except ValueError as e:
    print(f"Validation error: {e}")
    db.rollback()
except Exception as e:
    print(f"Unexpected error: {e}")
    db.rollback()
```

## Next Steps

After completing requirements analysis:
1. Review validation results
2. Address any missing items or warnings
3. Proceed to cloud provider recommendation (Task 4)
4. Generate migration plan (Task 6)
