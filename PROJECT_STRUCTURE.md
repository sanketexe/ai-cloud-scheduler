# Project Structure

```
TS_AI_CLOUD_SCHEDULER/
│
├── backend/                      # FastAPI Backend
│   ├── core/                    # Core business logic
│   │   ├── ai_assistant.py      # AI chat assistant
│   │   ├── auth.py              # Authentication & JWT
│   │   ├── aws_integration.py   # AWS Cost Explorer integration
│   │   ├── database.py          # Database configuration
│   │   ├── models.py            # SQLAlchemy models
│   │   └── ...
│   ├── alembic/                 # Database migrations
│   ├── main.py                  # FastAPI application entry
│   ├── celery_worker.py         # Background task worker
│   ├── celery_beat.py           # Task scheduler
│   └── requirements.txt         # Python dependencies
│
├── frontend/                     # React Frontend
│   ├── public/                  # Static assets
│   ├── src/
│   │   ├── components/          # Reusable components
│   │   │   ├── Layout/          # Header, Sidebar
│   │   │   └── MigrationWizard/ # Migration forms
│   │   ├── pages/               # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── CostAnalysis.tsx
│   │   │   ├── MigrationWizard.tsx
│   │   │   ├── MigrationDashboard.tsx
│   │   │   ├── ResourceOrganization.tsx
│   │   │   └── ...
│   │   ├── services/            # API clients
│   │   │   ├── api.ts
│   │   │   └── migrationApi.ts
│   │   ├── App.tsx              # Main app component
│   │   └── index.tsx            # Entry point
│   ├── package.json             # Node dependencies
│   └── Dockerfile               # Frontend container
│
├── monitoring/                   # Monitoring configurations
│   ├── prometheus.yml           # Prometheus config
│   ├── grafana/                 # Grafana dashboards
│   └── logstash/                # Logstash pipelines
│
├── scripts/                      # Utility scripts
│   ├── docker-health-check.sh   # Container health checks
│   └── ...
│
├── k8s/                         # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ...
│
├── .github/                     # GitHub workflows
│   └── workflows/
│
├── docker-compose.yml           # Production Docker config
├── docker-compose.override.yml  # Development overrides
├── docker-compose.prod.yml      # Production overrides
├── Dockerfile                   # Backend container image
│
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── config.example.json          # Configuration template
│
├── start-project.ps1            # Windows start script
├── stop-project.ps1             # Windows stop script
├── start-dev.py                 # Development start script
│
├── README.md                    # Main documentation
├── QUICK_START.md               # Quick start guide
├── SETUP_GUIDE.md               # Detailed setup guide
├── DEPLOYMENT_GUIDE.md          # Deployment instructions
├── CONTRIBUTING.md              # Contribution guidelines
└── LICENSE                      # License file
```

## Key Directories

### `/backend`
Python FastAPI backend with:
- REST API endpoints
- Cloud provider integrations (AWS, GCP, Azure)
- Authentication & authorization
- Background task processing (Celery)
- Database models & migrations

### `/frontend`
React TypeScript frontend with:
- Material-UI components
- Cost analysis dashboards
- Migration wizard
- Resource management UI
- Real-time charts (Recharts)

### `/monitoring`
Observability stack:
- Prometheus for metrics
- Grafana for visualization
- ELK stack for logging

### `/k8s`
Kubernetes deployment manifests for production

### `/scripts`
Utility scripts for development and operations

## Configuration Files

- **docker-compose.yml** - Main Docker configuration (15 services)
- **docker-compose.override.yml** - Development settings
- **docker-compose.prod.yml** - Production settings
- **.env** - Environment variables (not in git)
- **.env.example** - Environment template

## Entry Points

- **Backend API:** `backend/main.py`
- **Frontend:** `frontend/src/index.tsx`
- **Celery Worker:** `backend/celery_worker.py`
- **Celery Beat:** `backend/celery_beat.py`

## Docker Services

1. **postgres** - PostgreSQL database
2. **redis** - Redis cache
3. **api** - FastAPI backend
4. **frontend** - React app
5. **worker** - Celery worker
6. **scheduler** - Celery beat
7. **prometheus** - Metrics collection
8. **grafana** - Metrics visualization
9. **node-exporter** - System metrics
10. **cadvisor** - Container metrics
11. **elasticsearch** - Log storage
12. **logstash** - Log processing
13. **kibana** - Log visualization
14. **filebeat** - Log shipping

## Tech Stack

**Backend:**
- FastAPI (Python web framework)
- SQLAlchemy (ORM)
- Alembic (migrations)
- Celery (task queue)
- Redis (cache/broker)
- PostgreSQL (database)
- Boto3 (AWS SDK)

**Frontend:**
- React 18
- TypeScript
- Material-UI (MUI)
- React Router
- Recharts
- Axios

**Infrastructure:**
- Docker & Docker Compose
- Prometheus & Grafana
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Kubernetes (optional)

**Cloud Integrations:**
- AWS Cost Explorer
- GCP Billing API
- Azure Cost Management API
