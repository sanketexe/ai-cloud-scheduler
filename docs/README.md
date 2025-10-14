# Cloud Intelligence Platform

A comprehensive multi-cloud management system that combines intelligent workload scheduling, financial operations (FinOps) management, and resource health monitoring across AWS, GCP, Azure, and other cloud providers.

## 🚀 Features

- **Intelligent Workload Scheduling**: ML-powered scheduling across multiple cloud providers
- **FinOps Management**: Real-time cost tracking, budget management, and optimization recommendations
- **Performance Monitoring**: Comprehensive resource health monitoring and anomaly detection
- **Multi-Cloud Support**: Unified management across AWS, GCP, and Azure
- **Real-time Analytics**: Interactive dashboards and reporting
- **API-First Design**: RESTful APIs for integration with existing tools
- **Security & Compliance**: Enterprise-grade security with audit logging

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [User Guide](#user-guide)
- [Developer Guide](#developer-guide)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [Support](#support)

## 🏃 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Access to cloud provider accounts (AWS, GCP, Azure)

### Run with Docker Compose

```bash
# Clone the repository
git clone https://github.com/your-org/cloud-intelligence-platform.git
cd cloud-intelligence-platform

# Start the platform
docker-compose up -d

# Access the dashboard
open http://localhost:8501
```

### API Access

The REST API is available at `http://localhost:8000`

```bash
# Health check
curl http://localhost:8000/health

# Get sample workloads
curl http://localhost:8000/api/workloads/sample
```

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/cloud-intelligence-platform.git
   cd cloud-intelligence-platform
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API server**
   ```bash
   uvicorn api:app --reload
   ```

5. **Run the dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

### Production Deployment

See [Deployment Guide](deployment/README.md) for production deployment options:
- [Docker Deployment](deployment/docker.md)
- [Kubernetes Deployment](deployment/kubernetes.md)
- [Cloud Provider Deployment](deployment/cloud-providers.md)

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:password@localhost:5432/cloud_intelligence` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `INFLUXDB_URL` | InfluxDB connection string | `http://localhost:8086` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Environment name | `dev` |

### Cloud Provider Configuration

Configure cloud provider credentials:

```bash
# AWS
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# GCP
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-project-id

# Azure
export AZURE_CLIENT_ID=your_client_id
export AZURE_CLIENT_SECRET=your_client_secret
export AZURE_TENANT_ID=your_tenant_id
```

## 📚 Documentation

- [User Guide](docs/user-guide/README.md) - How to use the platform
- [API Documentation](docs/api/README.md) - REST API reference
- [Developer Guide](docs/developer-guide/README.md) - Development setup and guidelines
- [Deployment Guide](docs/deployment/README.md) - Production deployment
- [Operations Guide](docs/operations/README.md) - Monitoring and maintenance

## 🔧 Architecture

The platform consists of several key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │    REST API     │    │   CLI Tools     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌───▼────┐  ┌──────────┐  ┌─────▼─────┐  ┌──────────────┐   │
│Workload│  │  FinOps  │  │Performance│  │  Analytics   │   │
│Scheduler│  │  Engine  │  │ Monitor   │  │   & AI       │   │
└────────┘  └──────────┘  └───────────┘  └──────────────┘   │
    │            │              │               │            │
    └────────────┼──────────────┼───────────────┼────────────┘
                 │              │               │
         ┌───────▼──────────────▼───────────────▼───────┐
         │              Data Layer                      │
         │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
         │  │PostgreSQL│ │  Redis   │ │   InfluxDB   │ │
         │  └──────────┘ └──────────┘ └──────────────┘ │
         └──────────────────────────────────────────────┘
                                │
         ┌──────────────────────▼──────────────────────┐
         │           Cloud Provider APIs               │
         │  ┌──────┐    ┌──────┐    ┌──────────────┐  │
         │  │ AWS  │    │ GCP  │    │    Azure     │  │
         │  └──────┘    └──────┘    └──────────────┘  │
         └─────────────────────────────────────────────┘
```

## 🚀 Getting Started

### 1. Basic Workload Scheduling

```python
import requests

# Create a workload
workload = {
    "id": 1,
    "cpu_required": 2,
    "memory_required_gb": 4
}

# Schedule the workload
response = requests.post(
    "http://localhost:8000/api/simulation/run",
    json={
        "scheduler_type": "intelligent",
        "workloads": [workload]
    }
)

print(response.json())
```

### 2. Cost Tracking

```python
# Get cost data
response = requests.get("http://localhost:8000/api/costs/summary")
cost_data = response.json()

print(f"Total cost: ${cost_data['total_cost']}")
print(f"Cost by provider: {cost_data['by_provider']}")
```

### 3. Performance Monitoring

```python
# Get performance metrics
response = requests.get("http://localhost:8000/api/performance/metrics")
metrics = response.json()

print(f"CPU utilization: {metrics['cpu_utilization']}%")
print(f"Memory utilization: {metrics['memory_utilization']}%")
```

## 🔍 Monitoring

The platform includes comprehensive monitoring and observability:

- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert handling and notifications
- **Jaeger**: Distributed tracing
- **ELK Stack**: Centralized logging

Access monitoring tools:
- Grafana: http://localhost:3000 (admin/admin123)
- Prometheus: http://localhost:9090
- AlertManager: http://localhost:9093
- Kibana: http://localhost:5601

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# Load tests
pytest tests/load/ -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/cloud-intelligence-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/cloud-intelligence-platform/discussions)
- **Email**: support@cloud-intelligence.com

## 🗺️ Roadmap

- [ ] Multi-region deployment support
- [ ] Advanced ML models for cost prediction
- [ ] Integration with more cloud providers
- [ ] Mobile dashboard application
- [ ] Advanced security features

---

**Cloud Intelligence Platform** - Intelligent Multi-Cloud Management Made Simple