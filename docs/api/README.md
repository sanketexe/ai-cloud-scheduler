# API Documentation

The Cloud Intelligence Platform provides a comprehensive REST API for programmatic access to all platform features.

## Base URL

```
http://your-platform-url:8000
```

## Authentication

Currently, the API uses basic authentication. In production, implement proper authentication:

```bash
# Example with API key (when implemented)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/api/workloads
```

## API Reference

### Health Check

#### GET /health

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Providers

#### GET /api/providers/default

Get the default cloud providers configuration.

**Response:**
```json
[
  {
    "name": "AWS",
    "cpu_cost": 0.04,
    "memory_cost_gb": 0.01
  },
  {
    "name": "GCP",
    "cpu_cost": 0.035,
    "memory_cost_gb": 0.009
  },
  {
    "name": "Azure",
    "cpu_cost": 0.042,
    "memory_cost_gb": 0.011
  }
]
```

#### POST /api/providers

Create a new cloud provider configuration.

**Request Body:**
```json
{
  "name": "Custom Provider",
  "cpu_cost": 0.05,
  "memory_cost_gb": 0.012
}
```

**Response:**
```json
{
  "name": "Custom Provider",
  "cpu_cost": 0.05,
  "memory_cost_gb": 0.012,
  "status": "created"
}
```

### Virtual Machines

#### GET /api/vms/default

Get the default virtual machine configurations.

**Response:**
```json
[
  {
    "vm_id": 1,
    "cpu_capacity": 4,
    "memory_capacity_gb": 16,
    "provider": {
      "name": "AWS",
      "cpu_cost": 0.04,
      "memory_cost_gb": 0.01
    },
    "cpu_used": 0,
    "memory_used_gb": 0
  }
]
```

#### POST /api/vms

Create a new virtual machine configuration.

**Request Body:**
```json
{
  "vm_id": 5,
  "cpu_capacity": 8,
  "memory_capacity_gb": 32,
  "provider": {
    "name": "AWS",
    "cpu_cost": 0.04,
    "memory_cost_gb": 0.01
  }
}
```

### Workloads

#### GET /api/workloads/sample

Get sample workloads for testing.

**Response:**
```json
[
  {
    "id": 1,
    "cpu_required": 2,
    "memory_required_gb": 4
  },
  {
    "id": 2,
    "cpu_required": 1,
    "memory_required_gb": 2
  }
]
```

#### POST /api/workloads

Create a new workload.

**Request Body:**
```json
{
  "id": 100,
  "cpu_required": 2,
  "memory_required_gb": 4
}
```

**Response:**
```json
{
  "id": 100,
  "cpu_required": 2,
  "memory_required_gb": 4,
  "status": "created"
}
```

#### POST /api/workloads/generate

Generate random workloads for testing.

**Query Parameters:**
- `count` (integer): Number of workloads to generate (1-100)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/workloads/generate?count=5"
```

**Response:**
```json
[
  {
    "id": 1001,
    "cpu_required": 2,
    "memory_required_gb": 4
  },
  {
    "id": 1002,
    "cpu_required": 1,
    "memory_required_gb": 2
  }
]
```

#### POST /api/workloads/upload

Upload workloads from a CSV file.

**Request:**
- Content-Type: `multipart/form-data`
- File: CSV file with workload data

**CSV Format:**
```csv
workload_id,cpu_required,memory_required_gb
1,2,4
2,1,2
3,4,8
```

**Response:**
```json
{
  "workloads": [
    {
      "id": 1,
      "cpu_required": 2,
      "memory_required_gb": 4
    }
  ],
  "count": 1,
  "column_mapping": {
    "ID Column": "workload_id",
    "CPU Column": "cpu_required",
    "Memory Column": "memory_required_gb"
  }
}
```

#### POST /api/workloads/preview

Preview CSV file structure before uploading.

**Request:**
- Content-Type: `multipart/form-data`
- File: CSV file to preview

**Response:**
```json
{
  "filename": "workloads.csv",
  "columns": ["workload_id", "cpu_required", "memory_required_gb"],
  "sample_rows": [
    {
      "workload_id": "1",
      "cpu_required": "2",
      "memory_required_gb": "4"
    }
  ],
  "suggested_mapping": {
    "id_column": "workload_id",
    "cpu_column": "cpu_required",
    "memory_column": "memory_required_gb"
  },
  "mapping_confidence": 100
}
```

### Simulation

#### POST /api/simulation/run

Run a workload scheduling simulation.

**Request Body:**
```json
{
  "scheduler_type": "intelligent",
  "workloads": [
    {
      "id": 1,
      "cpu_required": 2,
      "memory_required_gb": 4
    },
    {
      "id": 2,
      "cpu_required": 1,
      "memory_required_gb": 2
    }
  ]
}
```

**Scheduler Types:**
- `random`: Random assignment
- `lowest_cost`: Cost-optimized assignment
- `round_robin`: Even distribution
- `intelligent`: ML-based optimization
- `hybrid`: Combined strategies

**Response:**
```json
{
  "simulation_id": "sim_123456",
  "scheduler_type": "intelligent",
  "results": {
    "total_workloads": 2,
    "successful_assignments": 2,
    "failed_assignments": 0,
    "success_rate": 100.0,
    "total_cost": 0.24,
    "average_assignment_time_ms": 15.5,
    "resource_utilization": {
      "cpu": 37.5,
      "memory": 37.5
    },
    "cost_breakdown": {
      "aws": 0.12,
      "gcp": 0.12,
      "azure": 0.0
    }
  },
  "assignments": [
    {
      "workload_id": 1,
      "vm_id": 2,
      "provider": "GCP",
      "success": true,
      "cost": 0.12
    },
    {
      "workload_id": 2,
      "vm_id": 1,
      "provider": "AWS",
      "success": true,
      "cost": 0.12
    }
  ]
}
```

### Cost Management

#### GET /api/costs/summary

Get cost summary across all providers.

**Query Parameters:**
- `period` (string): Time period (day, week, month, year)
- `provider` (string): Filter by provider (aws, gcp, azure)

**Response:**
```json
{
  "total_cost": 1250.75,
  "period": "month",
  "currency": "USD",
  "by_provider": {
    "aws": 650.25,
    "gcp": 400.50,
    "azure": 200.00
  },
  "by_service": {
    "compute": 800.00,
    "storage": 250.75,
    "networking": 200.00
  },
  "trends": {
    "daily_average": 40.35,
    "growth_rate": 5.2
  }
}
```

#### GET /api/costs/budgets

Get all budget configurations.

**Response:**
```json
[
  {
    "budget_id": "budget_001",
    "name": "Monthly Compute Budget",
    "amount": 1000.00,
    "period": "monthly",
    "current_spend": 750.25,
    "utilization": 75.0,
    "alert_thresholds": [80, 90, 100],
    "status": "on_track"
  }
]
```

#### POST /api/costs/budgets

Create a new budget.

**Request Body:**
```json
{
  "name": "Q1 Development Budget",
  "amount": 5000.00,
  "period": "quarterly",
  "alert_thresholds": [75, 90, 100],
  "scope": {
    "providers": ["aws", "gcp"],
    "services": ["compute", "storage"]
  }
}
```

### Performance Monitoring

#### GET /api/performance/metrics

Get current performance metrics.

**Query Parameters:**
- `resource_id` (string): Filter by resource ID
- `metric_type` (string): Filter by metric type
- `time_range` (string): Time range (1h, 24h, 7d, 30d)

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": [
    {
      "resource_id": "vm-001",
      "resource_type": "virtual_machine",
      "provider": "aws",
      "metrics": {
        "cpu_utilization": 65.5,
        "memory_utilization": 72.3,
        "disk_io": 1250,
        "network_io": 850
      }
    }
  ],
  "summary": {
    "average_cpu": 58.2,
    "average_memory": 64.7,
    "total_resources": 4
  }
}
```

#### GET /api/performance/anomalies

Get detected performance anomalies.

**Query Parameters:**
- `severity` (string): Filter by severity (low, medium, high, critical)
- `status` (string): Filter by status (open, investigating, resolved)

**Response:**
```json
[
  {
    "anomaly_id": "anom_001",
    "resource_id": "vm-002",
    "metric_type": "cpu_utilization",
    "severity": "high",
    "detected_at": "2024-01-15T09:45:00Z",
    "description": "CPU utilization spike detected",
    "current_value": 95.2,
    "expected_range": [20, 70],
    "suggested_actions": [
      "Scale up the instance",
      "Investigate running processes"
    ],
    "status": "open"
  }
]
```

### Analytics and Reporting

#### GET /api/analytics/dashboard

Get dashboard analytics data.

**Response:**
```json
{
  "overview": {
    "total_workloads": 1250,
    "active_workloads": 85,
    "total_cost_today": 125.75,
    "average_response_time": 245
  },
  "trends": {
    "workload_growth": 12.5,
    "cost_trend": -5.2,
    "performance_trend": 8.1
  },
  "alerts": {
    "critical": 0,
    "warning": 3,
    "info": 12
  }
}
```

#### POST /api/reports/generate

Generate a custom report.

**Request Body:**
```json
{
  "report_type": "cost_analysis",
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "filters": {
    "providers": ["aws", "gcp"],
    "services": ["compute"]
  },
  "format": "pdf"
}
```

**Response:**
```json
{
  "report_id": "report_001",
  "status": "generating",
  "estimated_completion": "2024-01-15T10:35:00Z",
  "download_url": "/api/reports/report_001/download"
}
```

## Error Handling

The API uses standard HTTP status codes and returns error details in JSON format.

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid workload configuration",
    "details": {
      "field": "cpu_required",
      "issue": "Must be a positive integer"
    }
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Rate Limit**: 1000 requests per hour per API key
- **Headers**: Rate limit information in response headers
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset time

## SDKs and Libraries

### Python SDK

```python
from cloud_intelligence import CloudIntelligenceClient

client = CloudIntelligenceClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Run simulation
result = client.simulation.run(
    scheduler_type="intelligent",
    workloads=[
        {"id": 1, "cpu_required": 2, "memory_required_gb": 4}
    ]
)

print(f"Success rate: {result.success_rate}%")
```

### JavaScript SDK

```javascript
import { CloudIntelligenceClient } from 'cloud-intelligence-js';

const client = new CloudIntelligenceClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your_api_key'
});

// Get cost summary
const costs = await client.costs.getSummary({
  period: 'month'
});

console.log(`Total cost: $${costs.total_cost}`);
```

## Webhooks

Configure webhooks to receive real-time notifications:

### Webhook Events

- `workload.scheduled`: Workload successfully scheduled
- `workload.failed`: Workload scheduling failed
- `cost.threshold_exceeded`: Budget threshold exceeded
- `performance.anomaly_detected`: Performance anomaly detected
- `system.health_check_failed`: System health check failed

### Webhook Configuration

```json
{
  "url": "https://your-app.com/webhooks/cloud-intelligence",
  "events": ["workload.scheduled", "cost.threshold_exceeded"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload

```json
{
  "event": "workload.scheduled",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "workload_id": 123,
    "vm_id": 2,
    "provider": "aws",
    "cost": 0.12
  }
}
```

## Examples

### Complete Workflow Example

```python
import requests
import json

base_url = "http://localhost:8000"

# 1. Check system health
health = requests.get(f"{base_url}/health")
print(f"System status: {health.json()['status']}")

# 2. Get available VMs
vms = requests.get(f"{base_url}/api/vms/default")
print(f"Available VMs: {len(vms.json())}")

# 3. Create workloads
workloads = [
    {"id": 1, "cpu_required": 2, "memory_required_gb": 4},
    {"id": 2, "cpu_required": 1, "memory_required_gb": 2}
]

# 4. Run simulation
simulation = requests.post(
    f"{base_url}/api/simulation/run",
    json={
        "scheduler_type": "intelligent",
        "workloads": workloads
    }
)

result = simulation.json()
print(f"Simulation success rate: {result['results']['success_rate']}%")
print(f"Total cost: ${result['results']['total_cost']}")

# 5. Get cost summary
costs = requests.get(f"{base_url}/api/costs/summary?period=day")
cost_data = costs.json()
print(f"Daily cost: ${cost_data['total_cost']}")
```

For more examples and detailed usage, see the [User Guide](../user-guide/README.md).