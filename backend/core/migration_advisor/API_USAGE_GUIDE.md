# Cloud Migration Advisor API Usage Guide

## Overview

This guide provides comprehensive examples and best practices for using the Cloud Migration Advisor API. The API follows RESTful principles and uses JSON for request and response payloads.

## Table of Contents

1. [Authentication](#authentication)
2. [Getting Started](#getting-started)
3. [Migration Workflow](#migration-workflow)
4. [Common Use Cases](#common-use-cases)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Best Practices](#best-practices)

## Authentication

All API endpoints require authentication using Bearer tokens (JWT).

### Obtaining a Token

```bash
curl -X POST https://api.cloudmigration.example.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your-username",
    "password": "your-password"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the Authorization header for all subsequent requests:

```bash
curl -X GET https://api.cloudmigration.example.com/v1/api/migrations/projects \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Getting Started

### Step 1: Create a Migration Project

```bash
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/projects \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "organization_name": "Acme Corporation"
  }'
```

Response:
```json
{
  "project_id": "mig-acme-corporation-20231116120000-abc123de",
  "project_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "organization_name": "Acme Corporation",
  "status": "assessment",
  "current_phase": "Initial Assessment",
  "estimated_completion": null,
  "created_at": "2023-11-16T12:00:00Z"
}
```

### Step 2: Create Organization Profile

```bash
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/mig-acme-corporation-20231116120000-abc123de/assessment/organization \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "company_size": "medium",
    "industry": "Financial Services",
    "current_infrastructure": "on_premises",
    "it_team_size": 15,
    "cloud_experience_level": "beginner",
    "geographic_presence": ["North America", "Europe"],
    "additional_context": {
      "compliance_requirements": ["SOC2", "GDPR"]
    }
  }'
```

Response:
```json
{
  "profile": {
    "company_size": "medium",
    "industry": "Financial Services",
    "current_infrastructure": "on_premises",
    "it_team_size": 15,
    "cloud_experience_level": "beginner",
    "geographic_presence": ["North America", "Europe"]
  },
  "timeline_estimation": {
    "estimated_days": 18,
    "estimated_completion_date": "2023-12-04T00:00:00Z",
    "breakdown": {
      "base_days": 14,
      "infrastructure_multiplier": 1.0,
      "experience_adjustment_days": 3,
      "team_size_factor": 1.0
    }
  }
}
```

## Migration Workflow

The typical migration workflow follows these phases:

1. **Assessment** - Create project and organization profile
2. **Analysis** - Define workloads and requirements
3. **Recommendation** - Generate provider recommendations
4. **Planning** - Create migration plan
5. **Execution** - Execute migration phases
6. **Integration** - Integrate with FinOps and capture baselines

### Complete Workflow Example

#### 1. Create Workload Profile

```bash
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/workloads \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workload_name": "Web Application",
    "application_type": "web",
    "total_compute_cores": 32,
    "total_memory_gb": 128,
    "total_storage_tb": 2.5,
    "database_types": ["postgresql", "redis"],
    "data_volume_tb": 1.2,
    "peak_transaction_rate": 5000
  }'
```

#### 2. Define Performance Requirements

```bash
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/performance-requirements \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "availability_target": 99.95,
    "latency_requirements": {
      "p50": 100,
      "p95": 250,
      "p99": 500
    },
    "disaster_recovery_rto": 60,
    "disaster_recovery_rpo": 15,
    "geographic_distribution": ["us-east-1", "eu-west-1"]
  }'
```

#### 3. Generate Provider Recommendations

```bash
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/recommendations/generate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "required_services": ["compute", "storage", "database", "ml", "analytics"],
    "target_monthly_budget": 50000.0,
    "compliance_requirements": ["GDPR", "SOC2", "HIPAA"],
    "source_infrastructure": "on_premises",
    "providers": ["aws", "gcp", "azure"]
  }'
```

Response:
```json
{
  "primary_recommendation": {
    "provider": "AWS",
    "rank": 1,
    "overall_score": 0.87,
    "confidence_score": 0.92,
    "justification": "AWS provides the best fit for your requirements...",
    "strengths": [
      "Comprehensive service catalog",
      "Strong compliance certifications",
      "Mature ML/AI services"
    ],
    "weaknesses": [
      "Slightly higher costs for compute",
      "Complex pricing model"
    ],
    "estimated_monthly_cost": 48500.0,
    "migration_duration_weeks": 12
  },
  "alternative_recommendations": [...],
  "overall_confidence": 0.89,
  "key_findings": [
    "All providers meet compliance requirements",
    "AWS offers best service availability",
    "GCP provides most cost-effective solution"
  ]
}
```

#### 4. Create Migration Plan

```bash
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/plan \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "target_provider": "aws",
    "migration_strategy": "phased",
    "target_start_date": "2024-01-15T00:00:00Z"
  }'
```

#### 5. Track Migration Progress

```bash
curl -X GET https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/plan/progress \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Response:
```json
{
  "plan_id": "plan-abc123",
  "overall_status": "in_progress",
  "total_phases": 5,
  "completed_phases": 2,
  "in_progress_phases": 1,
  "not_started_phases": 2,
  "failed_phases": 0,
  "overall_progress_percentage": 40.0,
  "current_phase": {
    "phase_id": "phase-3",
    "phase_name": "Database Migration",
    "status": "in_progress"
  },
  "days_elapsed": 15,
  "days_remaining": 25
}
```

## Common Use Cases

### Use Case 1: Discover and Organize Resources

After migration, discover all resources and organize them:

```bash
# Step 1: Discover resources
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/resources/discover \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "aws",
    "credentials": {
      "access_key_id": "YOUR_ACCESS_KEY",
      "secret_access_key": "YOUR_SECRET_KEY"
    },
    "regions": ["us-east-1", "us-west-2"],
    "resource_types": ["ec2", "s3", "rds", "lambda"]
  }'

# Step 2: Organize resources
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/resources/organize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "auto_categorize": true,
    "apply_tags": true
  }'
```

### Use Case 2: Generate Dimensional Views

View resources by different organizational dimensions:

```bash
# View by team
curl -X GET "https://api.cloudmigration.example.com/v1/api/resources/views/team?project_id=PROJECT_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"

# View by environment
curl -X GET "https://api.cloudmigration.example.com/v1/api/resources/views/environment?project_id=PROJECT_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"

# View by cost center
curl -X GET "https://api.cloudmigration.example.com/v1/api/resources/views/cost_center?project_id=PROJECT_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Use Case 3: Advanced Resource Filtering

Filter resources using complex expressions:

```bash
curl -X POST https://api.cloudmigration.example.com/v1/api/resources/filter \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "filter_expression": {
      "operator": "AND",
      "conditions": [
        {
          "field": "team",
          "operator": "equals",
          "value": "platform-engineering"
        },
        {
          "field": "environment",
          "operator": "in",
          "value": ["production", "staging"]
        },
        {
          "field": "resource_type",
          "operator": "not_equals",
          "value": "s3"
        }
      ]
    },
    "sort_by": "resource_name",
    "sort_order": "asc",
    "limit": 100
  }'
```

### Use Case 4: FinOps Integration

Integrate with FinOps platform after migration:

```bash
# Step 1: Integrate with FinOps
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/integration/finops \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "enable_cost_tracking": true,
    "enable_budget_alerts": true,
    "enable_waste_detection": true,
    "enable_optimization": true,
    "cost_allocation_method": "proportional"
  }'

# Step 2: Capture baseline metrics
curl -X POST https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/integration/baselines \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "capture_cost_data": true,
    "capture_performance_data": true,
    "capture_utilization_data": true,
    "baseline_period_days": 7
  }'

# Step 3: Generate migration report
curl -X GET https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/reports/final \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

### Error Response Format

```json
{
  "detail": "Detailed error message explaining what went wrong"
}
```

### Common Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `204 No Content` - Request successful, no content to return
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Missing or invalid authentication token
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Example Error Handling

```python
import requests

response = requests.post(
    'https://api.cloudmigration.example.com/v1/api/migrations/projects',
    headers={'Authorization': f'Bearer {token}'},
    json={'organization_name': 'Acme Corp'}
)

if response.status_code == 201:
    project = response.json()
    print(f"Project created: {project['project_id']}")
elif response.status_code == 400:
    error = response.json()
    print(f"Bad request: {error['detail']}")
elif response.status_code == 401:
    print("Authentication failed. Please check your token.")
else:
    print(f"Unexpected error: {response.status_code}")
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Rate Limit**: 1000 requests per hour per user
- **Burst Limit**: 100 requests per minute

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1699876800
```

When rate limit is exceeded, the API returns `429 Too Many Requests`.

## Best Practices

### 1. Use Pagination

Always use pagination for list endpoints to avoid large responses:

```bash
curl -X GET "https://api.cloudmigration.example.com/v1/api/migrations/projects?limit=50&offset=0" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 2. Store Project IDs

Store project IDs returned from the API for future reference rather than querying by organization name.

### 3. Handle Async Operations

Some operations (like resource discovery) may take time. Poll the status endpoint:

```bash
# Check assessment status
curl -X GET https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/assessment/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Validate Requirements

Use the validation endpoint before generating recommendations:

```bash
curl -X GET https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/requirements/validation \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 5. Secure Credentials

Never hardcode cloud provider credentials. Use environment variables or secure vaults:

```python
import os

credentials = {
    'access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
    'secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
}
```

### 6. Monitor Progress

Regularly check migration progress and phase status:

```bash
# Get progress every 5 minutes
while true; do
  curl -X GET https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/plan/progress \
    -H "Authorization: Bearer YOUR_TOKEN"
  sleep 300
done
```

### 7. Use Filters Efficiently

When querying large resource inventories, use filters to reduce response size:

```bash
curl -X GET "https://api.cloudmigration.example.com/v1/api/migrations/PROJECT_ID/resources?environment=production&team=platform&limit=50" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## SDK Examples

### Python SDK Example

```python
from migration_advisor_sdk import MigrationAdvisorClient

# Initialize client
client = MigrationAdvisorClient(
    api_url='https://api.cloudmigration.example.com/v1',
    token='YOUR_TOKEN'
)

# Create project
project = client.projects.create(organization_name='Acme Corp')

# Create organization profile
profile = client.assessment.create_organization_profile(
    project_id=project['project_id'],
    company_size='medium',
    industry='Financial Services',
    current_infrastructure='on_premises',
    it_team_size=15,
    cloud_experience_level='beginner'
)

# Generate recommendations
recommendations = client.recommendations.generate(
    project_id=project['project_id'],
    required_services=['compute', 'storage', 'database'],
    target_monthly_budget=50000.0,
    compliance_requirements=['GDPR', 'SOC2']
)

print(f"Primary recommendation: {recommendations['primary_recommendation']['provider']}")
```

### JavaScript SDK Example

```javascript
const { MigrationAdvisorClient } = require('@migration-advisor/sdk');

// Initialize client
const client = new MigrationAdvisorClient({
  apiUrl: 'https://api.cloudmigration.example.com/v1',
  token: 'YOUR_TOKEN'
});

// Create project
const project = await client.projects.create({
  organizationName: 'Acme Corp'
});

// Generate recommendations
const recommendations = await client.recommendations.generate({
  projectId: project.projectId,
  requiredServices: ['compute', 'storage', 'database'],
  targetMonthlyBudget: 50000.0,
  complianceRequirements: ['GDPR', 'SOC2']
});

console.log(`Primary recommendation: ${recommendations.primaryRecommendation.provider}`);
```

## Support

For additional support:

- **Documentation**: https://docs.cloudmigration.example.com
- **API Reference**: https://api.cloudmigration.example.com/docs
- **Support Email**: support@cloudmigration.example.com
- **Community Forum**: https://community.cloudmigration.example.com
