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
    from datetime import datetime, timedelta
    import json
    from pydantic import BaseModel
    from typing import Optional
    
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
        return [
            {
                "action_id": "stop_unused_instances",
                "action_type": "stop_unused_instances",
                "name": "Stop Unused EC2 Instances",
                "description": "Automatically stop instances with low utilization",
                "potential_savings": 500,
                "status": "enabled",
                "resource_type": "ec2_instance",
                "risk_level": "low",
                "estimated_execution_time": "5 minutes",
                "last_executed": "2024-12-16T10:30:00Z",
                "success_rate": 98.5
            },
            {
                "action_id": "resize_instances",
                "action_type": "resize_underutilized_instances",
                "name": "Resize Underutilized Instances", 
                "description": "Downsize instances based on usage patterns",
                "potential_savings": 300,
                "status": "enabled",
                "resource_type": "ec2_instance",
                "risk_level": "medium",
                "estimated_execution_time": "10 minutes",
                "last_executed": "2024-12-16T09:15:00Z",
                "success_rate": 95.2
            },
            {
                "action_id": "upgrade_gp2_to_gp3",
                "action_type": "upgrade_storage",
                "name": "Upgrade GP2 to GP3 Volumes",
                "description": "Upgrade EBS volumes for better performance and cost",
                "potential_savings": 200,
                "status": "enabled",
                "resource_type": "ebs_volume",
                "risk_level": "low",
                "estimated_execution_time": "3 minutes",
                "last_executed": "2024-12-16T11:45:00Z",
                "success_rate": 99.1
            },
            {
                "action_id": "release_unused_eips",
                "action_type": "release_elastic_ips",
                "name": "Release Unused Elastic IPs",
                "description": "Release unassociated Elastic IP addresses",
                "potential_savings": 150,
                "status": "enabled",
                "resource_type": "elastic_ip",
                "risk_level": "low",
                "estimated_execution_time": "2 minutes",
                "last_executed": "2024-12-16T08:20:00Z",
                "success_rate": 100.0
            },
            {
                "action_id": "delete_unattached_volumes",
                "action_type": "delete_volumes",
                "name": "Delete Unattached EBS Volumes",
                "description": "Remove volumes not attached to any instance",
                "potential_savings": 180,
                "status": "enabled",
                "resource_type": "ebs_volume",
                "risk_level": "medium",
                "estimated_execution_time": "5 minutes",
                "last_executed": "2024-12-16T07:30:00Z",
                "success_rate": 97.8
            }
        ]
    
    @app.get("/api/automation/stats")
    async def get_automation_stats():
        return {
            "total_actions": 15,
            "active_actions": 12,
            "monthly_savings": 2400,
            "actions_this_month": 45,
            "success_rate": 98.5,
            "automation_enabled": True,
            "last_execution": "2024-12-16T11:45:00Z",
            "pending_approvals": 3,
            "failed_actions": 1
        }
    
    @app.post("/api/automation/toggle")
    async def toggle_automation(request: dict):
        enabled = request.get("enabled", True)
        return {
            "success": True,
            "automation_enabled": enabled,
            "message": f"Automation {'enabled' if enabled else 'disabled'} successfully"
        }
    
    @app.post("/api/automation/actions/{action_id}/execute")
    async def execute_automation_action(action_id: str):
        return {
            "execution_id": f"exec_{action_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "action_id": action_id,
            "status": "started",
            "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "message": f"Action {action_id} execution started successfully"
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
    
    # Onboarding / Demo Mode endpoint
    class AWSCredentials(BaseModel):
        access_key_id: str
        secret_access_key: str
        region: str = "us-east-1"
        session_token: Optional[str] = None

    @app.post("/api/v1/onboarding/quick-setup")
    async def quick_setup_onboarding(creds: AWSCredentials):
        if creds.access_key_id == "DEMO" and creds.secret_access_key == "DEMO":
            return {
                "success": True,
                "message": "Demo mode activated. Loading sample data...",
                "account_id": "123456789012",
                "demo_mode": True
            }
        # For non-demo credentials, simulate a successful connection
        return {
            "success": True,
            "message": "Successfully connected to AWS. Initial scan started.",
            "account_id": "123456789012",
            "demo_mode": False
        }
    
    # ============================================================
    # Multi-Cloud API endpoints (for Multi-Cloud Dashboard & Migration Planner)
    # ============================================================

    @app.get("/api/v1/multi-cloud/providers")
    async def get_supported_providers():
        return [
            {
                "name": "Amazon Web Services",
                "provider_type": "aws",
                "supported_regions": ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"],
                "supported_services": ["EC2", "S3", "RDS", "Lambda", "EKS", "DynamoDB"],
                "pricing_model": "On-Demand / Reserved / Spot"
            },
            {
                "name": "Google Cloud Platform",
                "provider_type": "gcp",
                "supported_regions": ["us-central1", "us-east1", "europe-west1", "asia-south1"],
                "supported_services": ["Compute Engine", "Cloud Storage", "Cloud SQL", "Cloud Functions", "GKE", "Firestore"],
                "pricing_model": "On-Demand / Committed Use / Preemptible"
            },
            {
                "name": "Microsoft Azure",
                "provider_type": "azure",
                "supported_regions": ["eastus", "westus2", "westeurope", "centralindia"],
                "supported_services": ["Virtual Machines", "Blob Storage", "Azure SQL", "Azure Functions", "AKS", "Cosmos DB"],
                "pricing_model": "Pay-As-You-Go / Reserved / Spot"
            }
        ]

    # Sample on-premises workloads data
    sample_workloads = [
        {
            "id": "wl_001",
            "name": "Production Web Servers (2x Dell PowerEdge R750)",
            "description": "2 physical servers running Apache/Nginx, 64GB RAM, 16 cores each, 2TB SSD RAID",
            "created_at": "2025-01-15T10:00:00Z",
            "updated_at": "2025-02-20T14:30:00Z",
            "regions": ["on-premises"],
            "compliance_requirements": ["SOC2"]
        },
        {
            "id": "wl_002",
            "name": "Database Cluster (3x HP ProLiant DL380)",
            "description": "MySQL/PostgreSQL cluster, 128GB RAM each, 32 cores, 10TB storage, daily backups",
            "created_at": "2025-01-20T09:00:00Z",
            "updated_at": "2025-02-18T11:00:00Z",
            "regions": ["on-premises"],
            "compliance_requirements": ["HIPAA", "PCI-DSS"]
        },
        {
            "id": "wl_003",
            "name": "File Storage & NAS (Synology / NetApp)",
            "description": "50TB NAS for shared file storage, backups, and media assets",
            "created_at": "2025-02-01T08:00:00Z",
            "updated_at": "2025-02-25T16:00:00Z",
            "regions": ["on-premises"],
            "compliance_requirements": ["SOC2"]
        }
    ]

    @app.get("/api/v1/multi-cloud/workloads")
    async def get_workload_specifications(page: int = 1, page_size: int = 20):
        return {
            "workloads": sample_workloads,
            "total_count": len(sample_workloads),
            "page": page,
            "page_size": page_size
        }

    @app.post("/api/v1/multi-cloud/compare")
    async def compare_workload_costs(workload: dict):
        return {
            "id": f"cmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workload_id": workload.get("name", "unknown"),
            "comparison_date": datetime.utcnow().isoformat(),
            "aws_monthly_cost": 2450.00,
            "gcp_monthly_cost": 2180.00,
            "azure_monthly_cost": 2320.00,
            "aws_annual_cost": 29400.00,
            "gcp_annual_cost": 26160.00,
            "azure_annual_cost": 27840.00,
            "cost_breakdown": {
                "aws": {"compute": 1200, "storage": 400, "network": 250, "database": 350, "additional_services": 150, "support": 100, "total": 2450},
                "gcp": {"compute": 1050, "storage": 380, "network": 200, "database": 320, "additional_services": 130, "support": 100, "total": 2180},
                "azure": {"compute": 1150, "storage": 390, "network": 230, "database": 330, "additional_services": 120, "support": 100, "total": 2320}
            },
            "recommendations": [
                "GCP offers 11% savings over AWS for this workload",
                "Consider committed use discounts on GCP for additional 20% savings",
                "Azure Hybrid Benefit can reduce costs if you have existing Windows licenses"
            ],
            "lowest_cost_provider": "gcp",
            "cost_difference_percentage": {"aws_vs_gcp": 12.4, "aws_vs_azure": 5.6, "azure_vs_gcp": 6.4}
        }

    @app.post("/api/v1/multi-cloud/tco")
    async def calculate_tco(request: dict):
        years = request.get("time_horizon_years", 3)
        return {
            "id": f"tco_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workload_id": request.get("workload_id", "unknown"),
            "analysis_date": datetime.utcnow().isoformat(),
            "time_horizon_years": years,
            "aws_tco": {"year_1": 29400, "year_2": 28200, "year_3": 27000, "total": 84600},
            "gcp_tco": {"year_1": 26160, "year_2": 25000, "year_3": 24000, "total": 75160},
            "azure_tco": {"year_1": 27840, "year_2": 26700, "year_3": 25500, "total": 80040},
            "hidden_costs": {"migration": 5000, "training": 3000, "downtime": 2000, "compliance": 1500},
            "operational_costs": {"staff": 120000, "monitoring": 2400, "security": 3600},
            "cost_projections": {
                "aws": [29400, 28200, 27000],
                "gcp": [26160, 25000, 24000],
                "azure": [27840, 26700, 25500]
            },
            "total_tco_comparison": {"aws": 84600, "gcp": 75160, "azure": 80040},
            "recommended_provider": "gcp"
        }

    @app.post("/api/v1/multi-cloud/migration")
    async def analyze_migration(request: dict):
        source = request.get("source_provider", "on_premises")
        target = request.get("target_provider", "aws")
        target_name = {"aws": "AWS", "gcp": "GCP", "azure": "Azure"}.get(target, target.upper())
        return {
            "id": f"mig_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workload_id": request.get("workload_id", "unknown"),
            "source_provider": "on_premises",
            "target_provider": target,
            "analysis_date": datetime.utcnow().isoformat(),
            "migration_cost": 18500,
            "migration_timeline_days": 60,
            "break_even_months": 10,
            "cost_breakdown": {
                "cloud_infrastructure_setup": 2500,
                "data_migration_transfer": 5000,
                "application_re_architecture": 4500,
                "testing_and_validation": 3000,
                "team_training": 2000,
                "decommission_old_hardware": 1500
            },
            "risk_assessment": {
                "overall_risk_level": "medium",
                "technical_risks": [
                    "Data transfer bandwidth — large datasets may take days to upload",
                    "Application compatibility with cloud-native services",
                    "Network latency changes affecting end-user experience",
                    "Licensing changes for software running on cloud VMs"
                ],
                "business_risks": [
                    "Downtime during cutover window (estimated 2-4 hours)",
                    "Team needs cloud skills training (AWS/DevOps)",
                    "Ongoing cloud costs replace upfront hardware investment",
                    "Vendor lock-in considerations"
                ],
                "mitigation_strategies": [
                    "Use AWS Server Migration Service (SMS) for lift-and-shift",
                    "Run parallel environments for 2 weeks before cutover",
                    "Start with non-critical workloads to build confidence",
                    "Train team on AWS fundamentals before migration",
                    "Set up AWS Cost Explorer alerts from day one"
                ],
                "success_probability": 0.88
            },
            "recommendations": [
                f"Migrate to {target_name} using a phased approach over 8-10 weeks",
                "Use AWS Database Migration Service (DMS) for database migration",
                "Start with dev/staging servers, then move production last",
                "Set up VPN or AWS Direct Connect for secure data transfer",
                f"Estimated monthly cloud cost on {target_name}: $2,100 vs current on-prem TCO: $3,800/mo"
            ],
            "monthly_savings": 1700,
            "annual_savings": 20400,
            "roi_percentage": 110.3
        }

    @app.post("/api/v1/multi-cloud/validate")
    async def validate_workload(workload: dict):
        return {
            "is_valid": True,
            "errors": [],
            "warnings": ["Consider adding backup storage for disaster recovery"],
            "estimated_monthly_cost_range": {
                "min_cost": 1800,
                "max_cost": 3200,
                "currency": "USD"
            }
        }

    @app.get("/api/v1/multi-cloud/comparisons/{workload_id}")
    async def get_workload_comparisons(workload_id: str, page: int = 1, page_size: int = 10):
        return {
            "comparisons": [],
            "total_count": 0,
            "page": page,
            "page_size": page_size
        }

    @app.get("/api/v1/multi-cloud/services/{provider}")
    async def get_provider_services(provider: str, category: str = None):
        services = {
            "aws": [
                {"name": "EC2", "category": "compute", "description": "Virtual servers", "pricing_units": ["per hour"], "regions": ["us-east-1", "us-west-2"]},
                {"name": "S3", "category": "storage", "description": "Object storage", "pricing_units": ["per GB"], "regions": ["us-east-1", "us-west-2"]},
                {"name": "RDS", "category": "database", "description": "Managed databases", "pricing_units": ["per hour"], "regions": ["us-east-1", "us-west-2"]}
            ],
            "gcp": [
                {"name": "Compute Engine", "category": "compute", "description": "Virtual machines", "pricing_units": ["per hour"], "regions": ["us-central1", "us-east1"]},
                {"name": "Cloud Storage", "category": "storage", "description": "Object storage", "pricing_units": ["per GB"], "regions": ["us-central1", "us-east1"]},
                {"name": "Cloud SQL", "category": "database", "description": "Managed databases", "pricing_units": ["per hour"], "regions": ["us-central1", "us-east1"]}
            ],
            "azure": [
                {"name": "Virtual Machines", "category": "compute", "description": "Cloud VMs", "pricing_units": ["per hour"], "regions": ["eastus", "westus2"]},
                {"name": "Blob Storage", "category": "storage", "description": "Object storage", "pricing_units": ["per GB"], "regions": ["eastus", "westus2"]},
                {"name": "Azure SQL", "category": "database", "description": "Managed databases", "pricing_units": ["per hour"], "regions": ["eastus", "westus2"]}
            ]
        }
        return services.get(provider, [])

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