# Cloud Migration Advisor

## Overview

Intelligent cloud provider recommendation system supporting 5 major providers (AWS, Azure, GCP, IBM Cloud, Oracle Cloud) with weighted scoring across 12 dimensions.

**Version**: 2.0.0

---

## Features

- **5 Cloud Providers**: AWS ☁️, Azure 🔷, GCP 🌐, IBM 🔷, Oracle 🔴
- **Weighted Scoring**: 12 dimensions with multipliers (Compliance ×3.0, Workload ×2.5, etc.)
- **Hard Eliminators**: FedRAMP, HIPAA, budget, data residency
- **Real-Time Preview**: Live scoring as user completes assessment
- **Evidence-Based**: Transparent scoring breakdown
- **Complexity Assessment**: Automatic timeline estimation (LOW/MEDIUM/HIGH)

---

## Quick Start

### Access the Wizard
Navigate to: `http://localhost:3000/migration-wizard`

### Complete Assessment
1. Organization profile (company size, industry)
2. Workload details (types, tech stack)
3. Requirements (compliance, budget, performance)

### View Recommendation
- Top provider with score (0-100)
- Evidence breakdown by category
- Comparison matrix across providers
- Migration complexity and timeline

---

## Scoring Algorithm

### Category Weights

| Category | Multiplier | Impact |
|----------|-----------|--------|
| Compliance | ×3.0 | Highest |
| Workload Fit | ×2.5 | Very High |
| Tech Stack | ×2.0 | High |
| Budget | ×2.0 | High |
| AI/ML | ×1.5 | Medium |
| Scalability | ×1.5 | Medium |
| Data Residency | ×1.5 | Medium |
| Hybrid Cloud | ×1.2 | Medium |
| Support | ×1.2 | Medium |
| Migration Tools | ×1.0 | Low |
| Ecosystem | ×1.0 | Low |
| Innovation | ×0.8 | Low |

### Hard Eliminators

- **FedRAMP Required** → Only AWS and Azure
- **HIPAA with BAA** → All providers support
- **Budget Constraint** → Eliminates expensive options
- **Data Residency** → Must have presence in required regions

---

## Provider Profiles

### AWS ☁️
- **Best For**: Startups, e-commerce, general-purpose
- **Strengths**: Largest catalog (200+ services), global reach
- **Watch For**: Complex pricing, egress fees

### Azure 🔷
- **Best For**: Microsoft stack, enterprise, government
- **Strengths**: Microsoft integration, hybrid cloud, compliance
- **Watch For**: Portal complexity, premium support costs

### GCP 🌐
- **Best For**: AI/ML, analytics, cost optimization
- **Strengths**: AI/ML leadership, BigQuery, competitive pricing
- **Watch For**: Smaller partner network, fewer regions

### IBM Cloud 🔷
- **Best For**: IBM software, financial services, hybrid
- **Strengths**: IBM integration, Red Hat OpenShift, Watson AI
- **Watch For**: Narrower catalog, smaller community

### Oracle Cloud 🔴
- **Best For**: Oracle DB, ERP, database workloads
- **Strengths**: Oracle DB performance, BYOL, Autonomous DB
- **Watch For**: Limited appeal outside Oracle ecosystem

---

## API Endpoints

### Create Project
```http
POST /api/migration-advisor/projects
```

### Get Score Preview
```http
GET /api/migration-advisor/projects/{id}/score-preview
```

### Get Recommendation
```http
GET /api/migration-advisor/projects/{id}/enhanced-recommendation
```

Full API docs: See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

---

## Testing

```bash
# Run tests
python -m pytest backend/core/migration_advisor/test_enhanced_scoring.py -v

# Test coverage: 100% (scoring engine)
# Performance: < 50ms (P95)
```

---

## File Structure

```
backend/core/migration_advisor/
├── README.md                          # This file
├── API_DOCUMENTATION.md               # Complete API reference
├── enhanced_scoring_engine.py         # Core algorithm
├── test_enhanced_scoring.py           # Test suite
├── assessment_endpoints.py            # API endpoints
├── migration_complexity_calculator.py # Complexity assessment
├── provider_catalog.py                # Provider profiles
└── compliance_catalog.py              # Compliance frameworks
```

---

## Support

- **API Docs**: http://localhost:8000/docs
- **GitHub**: https://github.com/sanketexe/ai-cloud-scheduler

---

**Version**: 2.0.0  
**License**: MIT
