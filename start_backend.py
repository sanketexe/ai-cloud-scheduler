#!/usr/bin/env python3
"""
Start the FinOps Platform Backend API
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for development
os.environ.setdefault('ENVIRONMENT', 'development')
os.environ.setdefault('DEBUG', 'true')
os.environ.setdefault('DATABASE_URL', 'sqlite:///./finops_dev.db')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
os.environ.setdefault('JWT_SECRET_KEY', 'dev-secret-key-change-in-production')

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from datetime import datetime
    import json
    
    # Create a simplified FastAPI app for development
    app = FastAPI(
        title="FinOps Platform API",
        description="Enterprise Cloud Financial Operations Platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Sample data for the frontend
    sample_data = {
        "accounts": [
            {
                "id": "123456789012",
                "name": "Production Account",
                "provider": "aws",
                "region": "us-east-1",
                "monthly_cost": 15000,
                "potential_savings": 1200,
                "status": "active"
            },
            {
                "id": "123456789013", 
                "name": "Development Account",
                "provider": "aws",
                "region": "us-west-2",
                "monthly_cost": 8000,
                "potential_savings": 800,
                "status": "active"
            },
            {
                "id": "123456789014",
                "name": "Staging Account",
                "provider": "aws", 
                "region": "eu-west-1",
                "monthly_cost": 4000,
                "potential_savings": 400,
                "status": "active"
            }
        ],
        "cost_data": [
            {"date": "2024-12-01", "cost": 890, "savings": 45},
            {"date": "2024-12-02", "cost": 920, "savings": 38},
            {"date": "2024-12-03", "cost": 875, "savings": 52},
            {"date": "2024-12-04", "cost": 910, "savings": 41},
            {"date": "2024-12-05", "cost": 885, "savings": 48},
            {"date": "2024-12-06", "cost": 905, "savings": 43},
            {"date": "2024-12-07", "cost": 870, "savings": 55}
        ],
        "migration_assessments": [
            {
                "id": "assessment_001",
                "name": "Production Workload Migration",
                "source_provider": "aws",
                "target_providers": ["gcp", "azure"],
                "status": "completed",
                "created_at": "2024-12-10T10:00:00Z",
                "estimated_savings": 2400,
                "complexity": "medium"
            },
            {
                "id": "assessment_002", 
                "name": "Development Environment Migration",
                "source_provider": "aws",
                "target_providers": ["gcp"],
                "status": "in_progress",
                "created_at": "2024-12-12T14:30:00Z",
                "estimated_savings": 800,
                "complexity": "low"
            }
        ]
    }
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "FinOps Platform API",
            "version": "1.0.0",
            "status": "running",
            "features": [
                "Cost Analysis & Optimization",
                "Migration Planning & Assessment", 
                "Multi-Cloud Management",
                "Automated Cost Optimization",
                "Real-time Monitoring"
            ]
        }
    
    # Health check
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "api": "operational",
                "database": "operational",
                "cache": "operational"
            }
        }
    
    # Cost analysis endpoints
    @app.get("/api/cost-analysis")
    async def get_cost_analysis():
        return {
            "total_cost": sum(acc["monthly_cost"] for acc in sample_data["accounts"]),
            "potential_savings": sum(acc["potential_savings"] for acc in sample_data["accounts"]),
            "accounts": sample_data["accounts"],
            "cost_trend": sample_data["cost_data"]
        }
    
    # Migration endpoints
    @app.get("/api/migration/assessments")
    async def get_migration_assessments():
        return {
            "assessments": sample_data["migration_assessments"],
            "total": len(sample_data["migration_assessments"])
        }
    
    @app.post("/api/migration/assessments")
    async def create_migration_assessment(assessment_data: dict):
        new_assessment = {
            "id": f"assessment_{len(sample_data['migration_assessments']) + 1:03d}",
            "name": assessment_data.get("name", "New Assessment"),
            "source_provider": assessment_data.get("source_provider", "aws"),
            "target_providers": assessment_data.get("target_providers", ["gcp"]),
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "estimated_savings": assessment_data.get("estimated_savings", 0),
            "complexity": assessment_data.get("complexity", "medium")
        }
        sample_data["migration_assessments"].append(new_assessment)
        return new_assessment
    
    # Provider recommendations
    @app.get("/api/migration/recommendations")
    async def get_provider_recommendations():
        return {
            "recommendations": [
                {
                    "provider": "gcp",
                    "estimated_savings": 1200,
                    "migration_effort": "medium",
                    "compatibility_score": 85,
                    "benefits": ["Lower compute costs", "Better ML services", "Sustained use discounts"]
                },
                {
                    "provider": "azure",
                    "estimated_savings": 800,
                    "migration_effort": "low", 
                    "compatibility_score": 92,
                    "benefits": ["Hybrid cloud integration", "Enterprise features", "Windows workloads"]
                }
            ]
        }
    
    # Automation endpoints
    @app.get("/api/automation/actions")
    async def get_automation_actions():
        return {
            "actions": [
                {
                    "id": "stop_unused_instances",
                    "name": "Stop Unused EC2 Instances",
                    "description": "Automatically stop instances with low utilization",
                    "potential_savings": 500,
                    "status": "enabled"
                },
                {
                    "id": "resize_instances",
                    "name": "Resize Underutilized Instances", 
                    "description": "Downsize instances based on usage patterns",
                    "potential_savings": 300,
                    "status": "enabled"
                }
            ]
        }
    
    @app.get("/api/automation/stats")
    async def get_automation_stats():
        return {
            "total_actions": 15,
            "active_actions": 12,
            "monthly_savings": 2400,
            "actions_this_month": 45,
            "success_rate": 98.5
        }
    
    # Dashboard data
    @app.get("/api/dashboard")
    async def get_dashboard_data():
        total_cost = sum(acc["monthly_cost"] for acc in sample_data["accounts"])
        total_savings = sum(acc["potential_savings"] for acc in sample_data["accounts"])
        
        return {
            "overview": {
                "total_monthly_cost": total_cost,
                "potential_savings": total_savings,
                "savings_percentage": round((total_savings / total_cost) * 100, 1),
                "active_accounts": len(sample_data["accounts"])
            },
            "cost_trend": sample_data["cost_data"],
            "top_opportunities": [
                {"type": "Unused Instances", "savings": 800, "count": 12},
                {"type": "Storage Optimization", "savings": 400, "count": 25},
                {"type": "Reserved Instances", "savings": 600, "count": 8}
            ]
        }
    
    # Budget management
    @app.get("/api/budgets")
    async def get_budgets():
        return {
            "budgets": [
                {
                    "id": "budget_001",
                    "name": "Production Environment",
                    "amount": 12000,
                    "spent": 9500,
                    "remaining": 2500,
                    "period": "monthly"
                },
                {
                    "id": "budget_002",
                    "name": "Development Environment", 
                    "amount": 5000,
                    "spent": 4200,
                    "remaining": 800,
                    "period": "monthly"
                }
            ]
        }
    
    # Alerts
    @app.get("/api/alerts")
    async def get_alerts():
        return {
            "alerts": [
                {
                    "id": "alert_001",
                    "type": "budget_exceeded",
                    "message": "Development budget 84% utilized",
                    "severity": "warning",
                    "timestamp": "2024-12-16T10:30:00Z"
                },
                {
                    "id": "alert_002",
                    "type": "cost_spike",
                    "message": "Unusual cost increase detected in us-east-1",
                    "severity": "high",
                    "timestamp": "2024-12-16T09:15:00Z"
                }
            ]
        }
    
    # Reports
    @app.get("/api/reports")
    async def get_reports():
        return {
            "reports": [
                {
                    "id": "report_001",
                    "name": "Monthly Cost Analysis",
                    "type": "cost_analysis",
                    "generated_at": "2024-12-15T00:00:00Z",
                    "file_url": "/reports/monthly_cost_analysis.pdf"
                },
                {
                    "id": "report_002",
                    "name": "Migration Assessment Report",
                    "type": "migration_analysis", 
                    "generated_at": "2024-12-14T00:00:00Z",
                    "file_url": "/reports/migration_assessment.pdf"
                }
            ]
        }
    
    if __name__ == "__main__":
        print("🚀 Starting FinOps Platform Backend API...")
        print("📡 API Server: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("🎯 Frontend should connect automatically via proxy")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Installing required packages...")
    os.system("pip install fastapi uvicorn")
    print("🔄 Please run the script again after installation")