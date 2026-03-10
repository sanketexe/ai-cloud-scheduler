# Migration Advisor API Documentation

## Overview

The Migration Advisor API provides intelligent cloud provider recommendations using a sophisticated weighted scoring algorithm. This document covers the enhanced API endpoints for the 5-provider system (AWS, Azure, GCP, IBM Cloud, Oracle Cloud).

**Base URL**: `http://localhost:8000/api/migration-advisor`

**Version**: 2.0.0

**Last Updated**: March 9, 2026

---

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [Enhanced Scoring Endpoints](#enhanced-scoring-endpoints)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Rate Limits](#rate-limits)
7. [Examples](#examples)

---

## Authentication

All API requests require authentication using Bearer tokens.

```http
Authorization: Bearer <your_token_here>
```

**Getting a Token**:
```bash
POST /api/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "your_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## Core Endpoints

### 1. Create Migration Project

Create a new migration assessment project.

**Endpoint**: `POST /projects`

**Request Body**:
```json
{
  "organization_name": "Acme Corporation",
  "industry": "financial_services",
  "company_size": "enterprise"
}
```

**Response**: `201 Created`
```json
{
  "project_id": "proj_abc123xyz",
  "organization_name": "Acme Corporation",
  "created_at": "2026-03-09T10:30:00Z",
  "status": "in_progress"
}
```

---

### 2. Get Project Details

Retrieve details of a migration project.

**Endpoint**: `GET /projects/{project_id}`

**Response**: `200 OK`
```json
{
  "project_id": "proj_abc123xyz",
  "organization_name": "Acme Corporation",
  "industry": "financial_services",
  "company_size": "enterprise",
  "status": "in_progress",
  "completion_percentage": 65,
  "created_at": "2026-03-09T10:30:00Z",
  "updated_at": "2026-03-09T11:45:00Z"
}
```

---

### 3. Update Assessment Answers

Submit or update assessment answers.

**Endpoint**: `PUT /projects/{project_id}/answers`

**Request Body**:
```json
{
  "workload_types": ["web_applications", "databases", "analytics"],
  "tech_stack": {
    "languages": ["python", "javascript", "java"],
    "frameworks": ["react", "django", "spring"],
    "databases": ["postgresql", "mongodb"]
  },
  "compliance_requirements": ["soc2", "pci_dss", "gdpr"],
  "budget": {
    "monthly_budget": 50000,
    "cost_priority": "high"
  },
  "performance_requirements": {
    "availability_target": 99.99,
    "latency_p95_ms": 200
  }
}
```

**Response**: `200 OK`
```json
{
  "project_id": "proj_abc123xyz",
  "answers_updated": true,
  "completion_percentage": 85
}
```

---

## Enhanced Scoring Endpoints

### 4. Get Real-Time Score Preview

Get live scoring preview as the user completes the assessment.

**Endpoint**: `GET /projects/{project_id}/score-preview`

**Query Parameters**:
- `include_evidence` (optional): Include detailed scoring evidence (default: false)

**Response**: `200 OK`
```json
{
  "project_id": "proj_abc123xyz",
  "completion_percentage": 85,
  "scores": [
    {
      "provider": "azure",
      "score": 78.5,
      "rank": 1,
      "status": "recommended"
    },
    {
      "provider": "aws",
      "score": 72.3,
      "rank": 2,
      "status": "alternative"
    },
    {
      "provider": "ibm",
      "score": 68.1,
      "rank": 3,
      "status": "alternative"
    },
    {
      "provider": "gcp",
      "score": 52.4,
      "rank": 4,
      "status": "not_recommended"
    },
    {
      "provider": "oracle",
      "score": 35.2,
      "rank": 5,
      "status": "not_recommended"
    }
  ],
  "last_updated": "2026-03-09T11:45:00Z"
}
```

---

### 5. Get Enhanced Recommendation

Get the full recommendation with evidence, comparison matrix, and complexity assessment.

**Endpoint**: `GET /projects/{project_id}/enhanced-recommendation`

**Response**: `200 OK`
```json
{
  "project_id": "proj_abc123xyz",
  "recommended_provider": {
    "provider": "azure",
    "score": 78.5,
    "display_name": "Microsoft Azure",
    "icon": "🔷",
    "color": "#0078D4",
    "evidence": {
      "category_scores": {
        "compliance": 95.0,
        "workload_fit": 85.0,
        "tech_stack": 90.0,
        "budget": 70.0,
        "ai_ml": 75.0,
        "scalability": 80.0,
        "data_residency": 85.0,
        "hybrid_cloud": 95.0,
        "support_quality": 85.0,
        "migration_tools": 80.0,
        "ecosystem": 75.0,
        "innovation": 70.0
      },
      "weighted_contributions": {
        "compliance": 28.5,
        "workload_fit": 21.25,
        "tech_stack": 18.0,
        "budget": 14.0,
        "ai_ml": 11.25,
        "scalability": 12.0,
        "data_residency": 12.75,
        "hybrid_cloud": 11.4,
        "support_quality": 10.2,
        "migration_tools": 8.0,
        "ecosystem": 7.5,
        "innovation": 5.6
      },
      "strengths": [
        "Excellent compliance coverage with 90+ certifications",
        "Seamless Microsoft ecosystem integration (Office 365, AD)",
        "Strong hybrid cloud capabilities with Azure Arc",
        "Enterprise agreements provide cost flexibility",
        "Best-in-class Windows Server and .NET support"
      ],
      "best_for": [
        "Microsoft-centric technology stacks",
        "Enterprise organizations with EA agreements",
        "Government and regulated industries",
        "Hybrid cloud deployments"
      ],
      "watch_for": [
        "Azure Portal can be complex for beginners",
        "Premium support can be expensive",
        "Some services lag behind AWS in maturity"
      ]
    },
    "capability_scores": {
      "compute": 5,
      "storage": 5,
      "networking": 5,
      "databases": 4,
      "ai_ml": 4,
      "security": 5
    }
  },
  "alternatives": [
    {
      "provider": "aws",
      "score": 72.3,
      "display_name": "Amazon Web Services",
      "icon": "☁️",
      "color": "#FF9900",
      "reason": "Strong alternative with broader service catalog"
    },
    {
      "provider": "ibm",
      "score": 68.1,
      "display_name": "IBM Cloud",
      "icon": "🔷",
      "color": "#0F62FE",
      "reason": "Good fit for financial services and IBM software"
    }
  ],
  "comparison_matrix": {
    "providers": ["azure", "aws", "ibm", "gcp", "oracle"],
    "categories": [
      {
        "name": "Compliance",
        "scores": [95, 90, 85, 80, 70]
      },
      {
        "name": "Workload Fit",
        "scores": [85, 80, 75, 70, 60]
      },
      {
        "name": "Tech Stack",
        "scores": [90, 75, 70, 65, 55]
      },
      {
        "name": "Budget",
        "scores": [70, 65, 75, 80, 70]
      },
      {
        "name": "AI/ML",
        "scores": [75, 85, 70, 90, 50]
      },
      {
        "name": "Hybrid Cloud",
        "scores": [95, 80, 90, 70, 60]
      }
    ]
  },
  "migration_complexity": {
    "level": "MEDIUM",
    "timeline_weeks": "6-8",
    "factors": {
      "data_volume_tb": 25,
      "compliance_count": 3,
      "team_experience": "intermediate",
      "hybrid_required": true
    },
    "description": "Medium complexity migration requiring 6-8 weeks with experienced team"
  },
  "estimated_monthly_cost": {
    "azure": 48500,
    "aws": 52000,
    "ibm": 51000,
    "gcp": 45000,
    "oracle": 47000
  },
  "generated_at": "2026-03-09T11:45:00Z"
}
```

---

### 6. Get Recommendation Explanation

Get detailed explanation of why a specific provider was recommended.

**Endpoint**: `GET /projects/{project_id}/recommendation/explanation`

**Query Parameters**:
- `provider` (optional): Get explanation for specific provider (default: recommended provider)

**Response**: `200 OK`
```json
{
  "provider": "azure",
  "score": 78.5,
  "explanation": {
    "summary": "Azure is recommended due to excellent compliance coverage, Microsoft ecosystem integration, and strong hybrid cloud capabilities.",
    "key_factors": [
      {
        "category": "compliance",
        "score": 95.0,
        "weight": 3.0,
        "contribution": 28.5,
        "reason": "Azure offers 90+ compliance certifications including SOC2, PCI-DSS, and GDPR which match your requirements"
      },
      {
        "category": "workload_fit",
        "score": 85.0,
        "weight": 2.5,
        "contribution": 21.25,
        "reason": "Azure excels at enterprise applications and web workloads which align with your needs"
      },
      {
        "category": "tech_stack",
        "score": 90.0,
        "weight": 2.0,
        "contribution": 18.0,
        "reason": "Strong support for your Microsoft-centric stack including .NET and SQL Server"
      }
    ],
    "hard_eliminators_passed": [
      "budget_constraint",
      "data_residency",
      "compliance_requirements"
    ],
    "trade_offs": {
      "pros": [
        "Best Microsoft ecosystem integration",
        "Excellent hybrid cloud support",
        "Strong compliance coverage"
      ],
      "cons": [
        "Portal complexity for beginners",
        "Premium support costs",
        "Some service maturity gaps vs AWS"
      ]
    }
  }
}
```

---

## Data Models

### Provider

```typescript
interface Provider {
  provider: "aws" | "azure" | "gcp" | "ibm" | "oracle";
  display_name: string;
  icon: string;
  color: string;
  score: number;
  rank: number;
  status: "recommended" | "alternative" | "not_recommended" | "eliminated";
}
```

### Evidence

```typescript
interface Evidence {
  category_scores: Record<string, number>;
  weighted_contributions: Record<string, number>;
  strengths: string[];
  best_for: string[];
  watch_for: string[];
}
```

### MigrationComplexity

```typescript
interface MigrationComplexity {
  level: "LOW" | "MEDIUM" | "HIGH";
  timeline_weeks: string;
  factors: {
    data_volume_tb: number;
    compliance_count: number;
    team_experience: "none" | "beginner" | "intermediate" | "advanced";
    hybrid_required: boolean;
  };
  description: string;
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_PROJECT_ID",
    "message": "Project not found",
    "details": "No project exists with ID: proj_invalid123",
    "timestamp": "2026-03-09T11:45:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_PROJECT_ID` | 404 | Project not found |
| `INCOMPLETE_ASSESSMENT` | 400 | Assessment not complete enough for scoring |
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid authentication token |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks permission for this operation |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limits

- **Standard Tier**: 100 requests per minute
- **Premium Tier**: 1000 requests per minute

**Rate Limit Headers**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1678363200
```

---

## Examples

### Example 1: Complete Assessment Flow

```bash
# 1. Create project
curl -X POST http://localhost:8000/api/migration-advisor/projects \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "organization_name": "Acme Corp",
    "industry": "financial_services",
    "company_size": "enterprise"
  }'

# Response: {"project_id": "proj_abc123"}

# 2. Submit assessment answers
curl -X PUT http://localhost:8000/api/migration-advisor/projects/proj_abc123/answers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workload_types": ["web_applications", "databases"],
    "tech_stack": {"languages": ["python", "javascript"]},
    "compliance_requirements": ["soc2", "pci_dss"],
    "budget": {"monthly_budget": 50000, "cost_priority": "high"}
  }'

# 3. Get score preview
curl http://localhost:8000/api/migration-advisor/projects/proj_abc123/score-preview \
  -H "Authorization: Bearer $TOKEN"

# 4. Get full recommendation
curl http://localhost:8000/api/migration-advisor/projects/proj_abc123/enhanced-recommendation \
  -H "Authorization: Bearer $TOKEN"
```

### Example 2: Real-Time Score Updates

```javascript
// Frontend code for live score preview
const fetchScorePreview = async (projectId) => {
  const response = await fetch(
    `http://localhost:8000/api/migration-advisor/projects/${projectId}/score-preview`,
    {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );
  
  const data = await response.json();
  return data.scores;
};

// Poll every 2 seconds while user fills assessment
const interval = setInterval(async () => {
  const scores = await fetchScorePreview(projectId);
  updateScoreChart(scores);
}, 2000);
```

### Example 3: Provider Comparison

```python
import requests

# Get enhanced recommendation
response = requests.get(
    f"http://localhost:8000/api/migration-advisor/projects/{project_id}/enhanced-recommendation",
    headers={"Authorization": f"Bearer {token}"}
)

data = response.json()

# Extract comparison matrix
matrix = data["comparison_matrix"]
providers = matrix["providers"]
categories = matrix["categories"]

# Print comparison table
print(f"{'Category':<20}", end="")
for provider in providers:
    print(f"{provider:<10}", end="")
print()

for category in categories:
    print(f"{category['name']:<20}", end="")
    for score in category['scores']:
        print(f"{score:<10}", end="")
    print()
```

---

## Changelog

### Version 2.0.0 (March 2026)
- Added support for IBM Cloud and Oracle Cloud (5 providers total)
- Implemented weighted scoring algorithm with 12 dimensions
- Added hard eliminator logic
- Added real-time score preview endpoint
- Added migration complexity assessment
- Enhanced evidence generation with detailed breakdowns

### Version 1.0.0 (November 2023)
- Initial release with AWS, Azure, GCP support
- Basic scoring algorithm
- Core assessment endpoints

---

## Support

For API support:
- **Email**: api-support@cloudmigration.example.com
- **Documentation**: https://docs.cloudmigration.example.com
- **Status Page**: https://status.cloudmigration.example.com

---

**Version**: 2.0.0  
**Last Updated**: March 9, 2026
