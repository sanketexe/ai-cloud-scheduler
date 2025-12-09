# Cloud Intelligence Platform

A comprehensive FinOps platform for multi-cloud cost optimization, migration planning, and resource management.

## 🚀 Features

### FinOps Management
- **Real-time Cost Tracking** - Monitor cloud spending across AWS, GCP, and Azure
- **Budget Management** - Set budgets, track spending, and receive alerts
- **Cost Attribution** - Tag-based cost allocation by team, project, and environment
- **Waste Detection** - Identify unused resources and optimization opportunities
- **Reserved Instance Optimization** - Recommendations for RI purchases
- **Compliance & Governance** - Tagging policies and compliance monitoring

### Cloud Migration Advisor
- **Multi-Cloud Assessment** - Evaluate migration options across providers
- **Cost Comparison** - Compare costs between AWS, GCP, and Azure
- **Migration Planning** - Generate detailed migration plans with timelines
- **Resource Organization** - Categorize and organize cloud resources
- **Post-Migration Reports** - Comprehensive migration analysis and optimization

### AI Assistant
- **Context-Aware Help** - Get assistance during migration and cost analysis
- **Smart Suggestions** - Intelligent recommendations based on your data
- **Real-time Chat** - Interactive support throughout the platform

## 📋 Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Node.js 16+
- Cloud provider credentials (AWS, GCP, or Azure)

## 🏃 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd TS_AI_CLOUD_SCHEDULER
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start the Platform
```bash
docker-compose up -d
```

### 4. Access the Application
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Grafana:** http://localhost:3001 (monitoring)
- **Kibana:** http://localhost:5601 (logs)

## 🐳 Docker Services

The platform runs 15 containerized services:

### Core Services
- **PostgreSQL** - Main database (port 5432)
- **Redis** - Cache and message broker (port 6379)
- **Backend API** - FastAPI server (port 8000)
- **Frontend** - React application (port 3000)
- **Celery Worker** - Background task processor
- **Celery Beat** - Task scheduler

### Monitoring Stack
- **Prometheus** - Metrics collection (port 9090)
- **Grafana** - Metrics visualization (port 3001)
- **Node Exporter** - System metrics (port 9100)
- **cAdvisor** - Container metrics (port 8080)

### Logging Stack (ELK)
- **Elasticsearch** - Log storage (port 9200)
- **Logstash** - Log processing (ports 5044, 9600)
- **Kibana** - Log visualization (port 5601)
- **Filebeat** - Log shipping

## 📁 Project Structure

```
.
├── backend/              # FastAPI backend
│   ├── core/            # Core business logic
│   ├── alembic/         # Database migrations
│   └── main.py          # Application entry point
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # Reusable components
│   │   ├── pages/       # Page components
│   │   └── services/    # API services
│   └── public/
├── monitoring/          # Monitoring configurations
│   ├── prometheus.yml
│   ├── grafana/
│   └── logstash/
├── scripts/             # Utility scripts
├── k8s/                 # Kubernetes manifests
├── docker-compose.yml   # Production configuration
├── docker-compose.override.yml  # Development overrides
└── Dockerfile           # Backend container image
```

## 🔧 Development

### Start in Development Mode
```bash
docker-compose up
```

Development mode includes:
- Hot reload for backend and frontend
- Debug mode enabled
- Source code mounted as volumes
- Monitoring/logging disabled by default

### Run Backend Tests
```bash
cd backend
pytest
```

### Run Frontend Tests
```bash
cd frontend
npm test
```

### Database Migrations
```bash
# Create migration
docker-compose exec api alembic revision --autogenerate -m "description"

# Apply migrations
docker-compose exec api alembic upgrade head
```

## 🌐 API Documentation

Interactive API documentation is available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔐 Authentication

The platform uses JWT-based authentication:

1. Register a user via `/api/auth/register`
2. Login via `/api/auth/login` to get access token
3. Include token in Authorization header: `Bearer <token>`

## 📊 Monitoring & Logging

### Metrics (Grafana)
Access Grafana at http://localhost:3001
- Default credentials: admin/admin
- Pre-configured dashboards for system and application metrics

### Logs (Kibana)
Access Kibana at http://localhost:5601
- Search and analyze application logs
- View error tracking and audit trails

### Enable Monitoring in Development
```bash
docker-compose --profile monitoring up -d
```

### Enable Logging in Development
```bash
docker-compose --profile logging up -d
```

## 🚢 Production Deployment

### Using Docker Compose
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Using Kubernetes
```bash
kubectl apply -f k8s/
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 🔑 Environment Variables

Key environment variables (see `.env.example`):

```bash
# Database
DATABASE_URL=postgresql://finops:password@postgres:5432/finops_db

# Redis
REDIS_URL=redis://:password@redis:6379/0

# JWT
JWT_SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Cloud Providers
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
GCP_PROJECT_ID=your-project
AZURE_SUBSCRIPTION_ID=your-subscription

# OpenAI (for AI Assistant)
OPENAI_API_KEY=your-openai-key
```

## 🛠️ Common Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f api
docker-compose logs -f frontend

# Restart a service
docker-compose restart api

# Check service status
docker-compose ps

# Access database
docker-compose exec postgres psql -U finops -d finops_db

# Access Redis CLI
docker-compose exec redis redis-cli -a redis_password

# Run backend shell
docker-compose exec api python
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Check the [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. Review API documentation at http://localhost:8000/docs
4. Open an issue on GitHub

## 🎯 Roadmap

- [ ] Multi-region support
- [ ] Advanced ML-based cost predictions
- [ ] Mobile application
- [ ] Slack/Teams integrations
- [ ] Custom report builder
- [ ] API rate limiting and quotas

---

**Built with:** FastAPI, React, PostgreSQL, Redis, Celery, Docker, Prometheus, Grafana, ELK Stack
