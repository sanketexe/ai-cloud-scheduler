# ☁️ AI Cloud Scheduler — FinOps Platform

**An intelligent cloud financial operations (FinOps) platform that helps startups and SMBs optimize cloud costs, plan on-premises to cloud migrations, and automate infrastructure management using AI/ML.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB?logo=react)](frontend/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](backend/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](requirements.txt)

---

## 📌 What is this?

AI Cloud Scheduler is a **full-stack FinOps platform** designed for startups migrating from on-premises infrastructure to cloud providers (AWS, GCP, Azure). It provides:

- **Cost Analysis & Optimization** — Real-time AWS cost monitoring with anomaly detection
- **On-Prem → Cloud Migration Planner** — TCO comparison, risk assessment, and phased migration plans
- **Multi-Cloud Cost Comparison** — Side-by-side pricing across AWS, GCP, and Azure
- **AI-Powered Recommendations** — ML-based cost forecasting, anomaly alerts, and savings suggestions
- **Budget & Compliance Management** — Automated budget tracking with alert thresholds
- **Automation Engine** — Policy-based auto-scaling, scheduling, and remediation

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React 18)                      │
│                        localhost:3000                            │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │Dashboard │ │Migration │ │MultiCloud│ │  Cost Analysis     │  │
│  │          │ │ Planner  │ │Dashboard │ │  & Optimization    │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬───────────┘  │
│       │             │            │                 │              │
│  ┌────┴─────────────┴────────────┴─────────────────┴───────────┐ │
│  │              API Service Layer (Axios)                       │ │
│  │  api.ts | multiCloudApi.ts | migrationApi.ts | anomalyApi   │ │
│  └─────────────────────────┬───────────────────────────────────┘ │
└────────────────────────────┼─────────────────────────────────────┘
                             │  HTTP (REST API)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI + Python)                   │
│                        localhost:8000                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API Layer (api/)                        │   │
│  │  onboarding.py | multi_cloud.py | anomaly_detection.py    │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐   │
│  │                 Core Services (core/)                      │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  │   │
│  │  │ Cost Engine  │  │  Migration  │  │   AI / ML        │  │   │
│  │  │ & Optimizer  │  │   Advisor   │  │   Services       │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  │   │
│  │  │ Compliance  │  │ Automation  │  │   Budget &       │  │   │
│  │  │ Framework   │  │   Engine    │  │   Alerts         │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐   │
│  │              ML Pipeline (ml/)                             │   │
│  │  anomaly_detector | forecast_engine | cost_data_collector  │   │
│  │  lstm_detector | prophet_forecaster | feature_store        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Layer         | Technology                                                  |
|---------------|-------------------------------------------------------------|
| **Frontend**  | React 18, TypeScript, Material UI (MUI), React Query, Axios |
| **Backend**   | Python 3.10+, FastAPI, Uvicorn, Pydantic                    |
| **ML/AI**     | Scikit-learn, Prophet, LSTM (PyTorch), Isolation Forest      |
| **Database**  | PostgreSQL (prod), SQLite (dev), SQLAlchemy ORM              |
| **Caching**   | Redis                                                        |
| **Task Queue**| Celery + Redis                                               |
| **DevOps**    | Docker, Docker Compose, Kubernetes (k8s manifests)           |
| **Monitoring**| Prometheus, Grafana, InfluxDB (time-series)                  |

---

## 📂 Project Structure

```
TS_AI_CLOUD_SCHEDULER/
├── frontend/                    # React SPA
│   ├── src/
│   │   ├── pages/               # 26 page components
│   │   │   ├── Dashboard.tsx           # Main FinOps dashboard
│   │   │   ├── MigrationPlanner.tsx    # On-Prem → Cloud migration planner
│   │   │   ├── MigrationWizard.tsx     # Step-by-step migration wizard
│   │   │   ├── MultiCloudDashboard.tsx # AWS vs GCP vs Azure comparison
│   │   │   ├── CostAnalysis.tsx        # Detailed cost breakdown
│   │   │   ├── AWSCostAnalysis.tsx     # AWS-specific cost analysis
│   │   │   ├── AnomalyDashboard.tsx    # Cost anomaly detection
│   │   │   ├── AutomationDashboard.tsx # Automation rules & policies
│   │   │   ├── BudgetManagement.tsx    # Budget tracking & alerts
│   │   │   ├── Compliance.tsx          # Compliance monitoring
│   │   │   ├── Optimization.tsx        # Cost optimization suggestions
│   │   │   └── ...                     # Reports, Settings, Alerts, etc.
│   │   ├── components/          # Reusable UI components
│   │   │   ├── Layout/                 # Sidebar, Header
│   │   │   ├── Migration/              # Timeline, CostBenefit, RiskAssessment
│   │   │   ├── MigrationWizard/        # Multi-step wizard forms
│   │   │   ├── MultiCloud/             # CostMatrix, TCO, ProviderOverview
│   │   │   └── AI/                     # AI dashboard components
│   │   └── services/            # API service layer
│   │       ├── api.ts                  # Core API client
│   │       ├── multiCloudApi.ts        # Multi-cloud endpoints
│   │       ├── migrationApi.ts         # Migration endpoints
│   │       └── anomalyApi.ts           # Anomaly detection endpoints
│   └── package.json
│
├── backend/                     # FastAPI backend
│   ├── main.py                  # App entry point & route registration
│   ├── api/                     # REST API endpoints
│   │   ├── onboarding.py               # AWS credential setup & demo mode
│   │   ├── multi_cloud.py              # Multi-cloud comparison API
│   │   ├── anomaly_detection.py        # Anomaly detection API
│   │   └── multi_cloud_models.py       # Pydantic models
│   ├── core/                    # Business logic (146 modules)
│   │   ├── finops_engine.py            # Core FinOps cost engine
│   │   ├── aws_cost_analyzer.py        # AWS cost analysis
│   │   ├── cost_anomaly_detector.py    # Cost anomaly detection
│   │   ├── migration_advisor/          # Migration planning (69 files)
│   │   ├── tco_calculator.py           # Total Cost of Ownership
│   │   ├── multi_cloud_cost_engine.py  # Multi-cloud pricing
│   │   ├── automation_endpoints.py     # Automation rules
│   │   ├── compliance_manager.py       # Compliance framework
│   │   ├── budget_management_system.py # Budget tracking
│   │   ├── ai_orchestrator.py          # AI service orchestration
│   │   ├── policy_manager.py           # Policy engine
│   │   └── ...                         # 130+ more modules
│   ├── ml/                      # Machine Learning pipeline
│   │   ├── anomaly_detector.py         # Anomaly detection models
│   │   ├── forecast_engine.py          # Cost forecasting
│   │   ├── prophet_forecaster.py       # Facebook Prophet integration
│   │   ├── lstm_anomaly_detector.py    # LSTM neural network
│   │   ├── training_pipeline.py        # Model training
│   │   ├── feature_store.py            # ML feature engineering
│   │   └── ...                         # 20+ ML modules
│   └── requirements.txt
│
├── start_backend.py             # Simplified dev server (mock data)
├── docker-compose.yml           # Full-stack Docker setup
├── k8s/                         # Kubernetes deployment manifests
├── monitoring/                  # Prometheus & Grafana configs
├── docs/                        # Documentation
└── scripts/                     # Utility scripts
```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 16+ and **npm**
- **Python** 3.10+
- **pip** (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/sanketexe/ai-cloud-scheduler.git
cd ai-cloud-scheduler
```

### 2. Install Dependencies

```bash
# Backend
pip install fastapi uvicorn pydantic

# Frontend
cd frontend
npm install
cd ..
```

### 3. Start the Application

**Terminal 1 — Backend API:**
```bash
python start_backend.py
```
> Backend runs at **http://localhost:8000**
> API docs available at **http://localhost:8000/docs**

**Terminal 2 — Frontend:**
```bash
cd frontend
npm start
```
> Frontend runs at **http://localhost:3000**

### 4. Login

Open **http://localhost:3000** and click **"Try Demo Mode"** to explore the platform with sample data — no AWS credentials needed.

---

## 📋 Key Features

### 1. On-Premises → Cloud Migration Planner
Plan your migration from physical servers to AWS/cloud:
- Select on-prem workloads (e.g., Dell PowerEdge, HP ProLiant servers)
- Choose target cloud provider (AWS, GCP, Azure)
- Get migration cost breakdown, timeline (in days), and ROI analysis
- Risk assessment with mitigation strategies
- Phased migration process: Infrastructure Audit → TCO Comparison → Risk Assessment → Timeline → Training → Go-Live

### 2. Multi-Cloud Cost Comparison
- Compare pricing across **AWS**, **GCP**, and **Azure** for identical workloads
- Cost breakdown by category: compute, storage, network, database
- TCO analysis over 1-5 year time horizons
- Savings recommendations with provider-specific tips

### 3. Cost Analysis & Anomaly Detection
- Real-time AWS cost monitoring and trending
- ML-powered anomaly detection (LSTM, Isolation Forest)
- Cost forecasting with Facebook Prophet
- Automated alerts for cost spikes

### 4. Budget Management
- Set monthly/quarterly budget limits per team or project
- Track budget utilization in real-time
- Customizable alert thresholds (50%, 80%, 100%)
- Budget forecasting and trend analysis

### 5. Automation & Optimization
- Policy-based cost optimization rules
- Auto-scaling recommendations for EC2, RDS, EKS
- Scheduled resource start/stop for dev environments
- Waste detection for idle/underutilized resources

### 6. Compliance & Governance
- Built-in compliance frameworks (SOC2, HIPAA, PCI-DSS, GDPR)
- Tagging policy enforcement
- Resource organization and taxonomy management
- Audit logging and reporting

---

## 🐳 Docker Deployment

For a full production-like deployment with all services:

```bash
docker-compose up -d
```

This starts:
- FastAPI backend with PostgreSQL
- React frontend (Nginx)
- Redis (caching & task queue)
- Celery workers (background jobs)
- Prometheus + Grafana (monitoring)

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/onboarding/quick-setup` | Connect AWS credentials or start demo mode |
| `GET`  | `/api/v1/multi-cloud/providers` | List supported cloud providers |
| `GET`  | `/api/v1/multi-cloud/workloads` | Get on-premises workload inventory |
| `POST` | `/api/v1/multi-cloud/migration` | Analyze on-prem → cloud migration |
| `POST` | `/api/v1/multi-cloud/compare` | Compare workload costs across providers |
| `POST` | `/api/v1/multi-cloud/tco` | Calculate Total Cost of Ownership |
| `GET`  | `/api/cost-analysis` | AWS cost analysis data |
| `GET`  | `/api/dashboard` | Dashboard overview metrics |
| `GET`  | `/api/budgets` | Budget tracking data |
| `GET`  | `/api/alerts` | Cost alerts and notifications |
| `GET`  | `/health` | Health check |

Full API documentation: **http://localhost:8000/docs** (Swagger UI)

---

## 🤖 AI/ML Capabilities

The platform includes a comprehensive ML pipeline for intelligent cost management:

| Model | Purpose | Module |
|-------|---------|--------|
| **LSTM Neural Network** | Time-series anomaly detection | `ml/lstm_anomaly_detector.py` |
| **Isolation Forest** | Statistical anomaly detection | `ml/isolation_forest_detector.py` |
| **Facebook Prophet** | Cost forecasting & seasonality | `ml/prophet_forecaster.py` |
| **Ensemble Scorer** | Combined anomaly confidence scoring | `ml/ensemble_scorer.py` |
| **Feature Store** | Automated feature engineering | `ml/feature_store.py` |
| **Training Pipeline** | Automated model retraining | `ml/training_pipeline.py` |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.