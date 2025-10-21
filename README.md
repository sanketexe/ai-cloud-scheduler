# FinOps Platform

A comprehensive Financial Operations (FinOps) platform for cloud cost management, optimization, and governance. Built with React TypeScript frontend and Python FastAPI backend.

## 🚀 Features

- **Cost Analysis**: Multi-cloud cost tracking and attribution
- **Budget Management**: Budget creation, monitoring, and alerts
- **Waste Detection**: Identify underutilized resources
- **RI Optimization**: Reserved Instance recommendations
- **Compliance**: Tagging policies and governance
- **Interactive Dashboard**: Real-time cost visualization

## 🏗️ Architecture

```
finops-platform/
├── frontend/          # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
├── backend/           # Python FastAPI backend
│   ├── core/         # Core FinOps modules
│   ├── main.py       # API entry point
│   └── finops_api.py # API endpoints
└── README.md
```

## 🛠️ Tech Stack

**Frontend:**
- React 18 with TypeScript
- Material-UI for components
- Recharts for data visualization
- React Router for navigation

**Backend:**
- Python 3.10+ with FastAPI
- Pydantic for data validation
- Pandas for data processing
- NumPy for calculations

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Start the entire platform
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

**Backend:**
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python start-dev.py
```

**Frontend:**
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Access the Application

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📊 Core Modules

- **Cost Attribution Engine**: Tag-based cost allocation and chargeback
- **Budget Management System**: Budget lifecycle and alert management
- **Waste Detection Engine**: Resource optimization recommendations
- **RI Optimization System**: Reserved Instance analysis
- **Compliance Framework**: Tagging policies and governance
- **Cloud API Integration**: Multi-cloud provider connectivity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.