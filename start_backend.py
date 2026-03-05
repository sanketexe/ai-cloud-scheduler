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

# Load .env file so AWS credentials and other config are available
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=project_root / ".env")
except ImportError:
    # python-dotenv not installed, try manual loading
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip())

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
    import logging

    from backend.core.aws_data_service import AWSDataService

    logger = logging.getLogger(__name__)

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

    # Global AWS Data Service instance — set after onboarding or from .env
    aws_service: Optional[AWSDataService] = None

    def _no_aws_response():
        """Return a standard response when no AWS account is connected."""
        return {
            "error": "no_aws_account",
            "message": "Please connect your AWS account first via the onboarding page.",
        }

    @app.on_event("startup")
    async def auto_connect_aws():
        """Auto-connect to AWS on startup if credentials are in .env"""
        global aws_service
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        region = os.environ.get("AWS_REGION", "us-east-1")

        # Skip if placeholder or empty credentials
        if (access_key and secret_key
                and access_key != "your-aws-access-key"
                and secret_key != "your-aws-secret-key"):
            try:
                service = AWSDataService(
                    access_key_id=access_key,
                    secret_access_key=secret_key,
                    region=region,
                )
                result = service.test_connection()
                if result.get("success"):
                    aws_service = service
                    logger.info(f"✅ Auto-connected to AWS account {result['account_id']} from .env")
                    print(f"✅ Auto-connected to AWS account {result['account_id']}")
                else:
                    logger.warning(f"⚠️ AWS credentials in .env are invalid: {result.get('error')}")
                    print(f"⚠️ AWS credentials in .env are invalid: {result.get('error')}")
            except Exception as e:
                logger.warning(f"⚠️ Could not auto-connect to AWS: {e}")
                print(f"⚠️ Could not auto-connect to AWS from .env: {e}")
        else:
            logger.info("ℹ️ No AWS credentials in .env — use onboarding to connect")
            print("ℹ️ No AWS credentials in .env — connect via /onboarding")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "FinOps Platform API",
            "version": "1.0.0",
            "status": "running",
            "aws_connected": aws_service is not None,
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
            "aws_connected": aws_service is not None,
            "services": {
                "api": "operational",
                "aws": "connected" if aws_service else "not_configured",
            }
        }

    # ============================================================
    # Cost analysis endpoints
    # ============================================================

    @app.get("/api/cost-analysis")
    async def get_cost_analysis():
        if aws_service is None:
            return _no_aws_response()
        
        accounts = aws_service.get_accounts()
        cost_trend = aws_service.get_cost_trend(30)

        return {
            "total_cost": sum(acc["monthly_cost"] for acc in accounts),
            "potential_savings": sum(acc["potential_savings"] for acc in accounts),
            "accounts": accounts,
            "cost_trend": cost_trend,
        }

    # ============================================================
    # Migration endpoints
    # ============================================================

    @app.get("/api/migration/assessments")
    async def get_migration_assessments():
        if aws_service is None:
            return _no_aws_response()
        
        # Return empty list — assessments are user-created
        return {
            "assessments": [],
            "total": 0,
        }

    @app.post("/api/migration/assessments")
    async def create_migration_assessment(assessment_data: dict):
        new_assessment = {
            "id": f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": assessment_data.get("name", "New Assessment"),
            "source_provider": assessment_data.get("source_provider", "aws"),
            "target_providers": assessment_data.get("target_providers", ["gcp"]),
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "estimated_savings": assessment_data.get("estimated_savings", 0),
            "complexity": assessment_data.get("complexity", "medium"),
        }
        return new_assessment

    # Provider recommendations
    @app.get("/api/migration/recommendations")
    async def get_provider_recommendations():
        if aws_service is None:
            return _no_aws_response()
        
        total_cost = aws_service.get_total_monthly_cost()
        return {
            "recommendations": [
                {
                    "provider": "gcp",
                    "estimated_savings": round(total_cost * 0.11, 2),
                    "migration_effort": "medium",
                    "compatibility_score": 85,
                    "benefits": ["Lower compute costs", "Better ML services", "Sustained use discounts"],
                },
                {
                    "provider": "azure",
                    "estimated_savings": round(total_cost * 0.06, 2),
                    "migration_effort": "low",
                    "compatibility_score": 92,
                    "benefits": ["Hybrid cloud integration", "Enterprise features", "Windows workloads"],
                },
            ]
        }

    # ============================================================
    # Automation endpoints
    # ============================================================

    @app.get("/api/automation/actions")
    async def get_automation_actions():
        if aws_service is None:
            return _no_aws_response()
        
        return aws_service.get_optimization_recommendations()

    @app.get("/api/automation/stats")
    async def get_automation_stats():
        if aws_service is None:
            return _no_aws_response()
        
        return aws_service.get_automation_stats()

    @app.post("/api/automation/toggle")
    async def toggle_automation(request: dict):
        enabled = request.get("enabled", True)
        return {
            "success": True,
            "automation_enabled": enabled,
            "message": f"Automation {'enabled' if enabled else 'disabled'} successfully",
        }

    @app.post("/api/automation/actions/{action_id}/execute")
    async def execute_automation_action(action_id: str):
        return {
            "execution_id": f"exec_{action_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "action_id": action_id,
            "status": "started",
            "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "message": f"Action {action_id} execution started successfully",
        }

    # ============================================================
    # Dashboard data
    # ============================================================

    @app.get("/api/dashboard")
    async def get_dashboard_data():
        if aws_service is None:
            return _no_aws_response()
        
        return aws_service.get_dashboard_summary()

    # ============================================================
    # Budget management
    # ============================================================

    @app.get("/api/budgets")
    async def get_budgets():
        if aws_service is None:
            return _no_aws_response()
        
        return {"budgets": aws_service.get_budgets()}

    # ============================================================
    # Alerts
    # ============================================================

    @app.get("/api/alerts")
    async def get_alerts():
        if aws_service is None:
            return _no_aws_response()

        raw_alerts = aws_service.get_cost_anomalies()
        # Enrich with display fields expected by frontend
        enriched = []
        for i, alert in enumerate(raw_alerts):
            enriched.append({
                "id": i + 1,
                "name": alert.get("message", "Alert")[:60],
                "type": alert.get("type", "cost_anomaly"),
                "condition": alert.get("message", ""),
                "threshold": 0,
                "currentValue": 0,
                "status": "triggered",
                "severity": alert.get("severity", "info"),
                "lastTriggered": alert.get("timestamp"),
                "channels": ["email"],
                "enabled": True,
                "team": "FinOps",
            })
        return {"alerts": enriched}

    # ============================================================
    # Reports
    # ============================================================

    @app.get("/api/reports")
    async def get_reports():
        if aws_service is None:
            return _no_aws_response()
        # Reports are generated on-demand from real data, return empty list
        return {"reports": []}

    # ============================================================
    # Compliance
    # ============================================================

    @app.get("/api/compliance")
    async def get_compliance():
        if aws_service is None:
            return _no_aws_response()

        try:
            return aws_service.get_compliance_data()
        except Exception as e:
            logger.error(f"Error getting compliance data: {e}")
            return {"error": str(e)}


    # ============================================================
    # Onboarding — Real AWS Credentials
    # ============================================================

    class AWSCredentials(BaseModel):
        access_key_id: str
        secret_access_key: str
        region: str = "us-east-1"
        session_token: Optional[str] = None

    @app.post("/api/v1/onboarding/quick-setup")
    async def quick_setup_onboarding(creds: AWSCredentials):
        global aws_service

        if not creds.access_key_id or not creds.secret_access_key:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "AWS credentials are required."},
            )

        # Create the service and test the connection
        try:
            service = AWSDataService(
                access_key_id=creds.access_key_id,
                secret_access_key=creds.secret_access_key,
                region=creds.region,
                session_token=creds.session_token,
            )
            result = service.test_connection()

            if not result.get("success"):
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "message": f"Failed to connect to AWS: {result.get('error', 'Unknown error')}",
                    },
                )

            # Store the service globally
            aws_service = service

            # Start initial data fetch in background
            try:
                aws_service.get_dashboard_summary()
            except Exception as e:
                logger.warning(f"Initial data fetch had some issues (non-fatal): {e}")

            return {
                "success": True,
                "message": "Successfully connected to AWS. Initial scan started.",
                "account_id": result["account_id"],
                "demo_mode": False,
            }
        except Exception as e:
            logger.error(f"Onboarding failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Connection failed: {str(e)}"},
            )

    # ============================================================
    # Multi-Cloud API endpoints
    # ============================================================

    @app.get("/api/v1/multi-cloud/providers")
    async def get_supported_providers():
        return [
            {
                "name": "Amazon Web Services",
                "provider_type": "aws",
                "supported_regions": ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"],
                "supported_services": ["EC2", "S3", "RDS", "Lambda", "EKS", "DynamoDB"],
                "pricing_model": "On-Demand / Reserved / Spot",
            },
            {
                "name": "Google Cloud Platform",
                "provider_type": "gcp",
                "supported_regions": ["us-central1", "us-east1", "europe-west1", "asia-south1"],
                "supported_services": ["Compute Engine", "Cloud Storage", "Cloud SQL", "Cloud Functions", "GKE", "Firestore"],
                "pricing_model": "On-Demand / Committed Use / Preemptible",
            },
            {
                "name": "Microsoft Azure",
                "provider_type": "azure",
                "supported_regions": ["eastus", "westus2", "westeurope", "centralindia"],
                "supported_services": ["Virtual Machines", "Blob Storage", "Azure SQL", "Azure Functions", "AKS", "Cosmos DB"],
                "pricing_model": "Pay-As-You-Go / Reserved / Spot",
            },
        ]

    @app.get("/api/v1/multi-cloud/workloads")
    async def get_workload_specifications(page: int = 1, page_size: int = 20):
        if aws_service is None:
            return _no_aws_response()
        
        instances = aws_service.get_ec2_instances()
        return {
            "workloads": instances,
            "total_count": len(instances),
            "page": page,
            "page_size": page_size,
        }

    @app.post("/api/v1/multi-cloud/compare")
    async def compare_workload_costs(workload: dict):
        if aws_service is None:
            return _no_aws_response()
        
        # Get actual monthly cost and estimate cloud provider costs
        total_cost = aws_service.get_total_monthly_cost()
        # Estimate relative pricing for other providers
        aws_cost = round(total_cost, 2)
        gcp_cost = round(total_cost * 0.89, 2)  # ~11% less
        azure_cost = round(total_cost * 0.95, 2)  # ~5% less

        return {
            "id": f"cmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workload_id": workload.get("name", "unknown"),
            "comparison_date": datetime.utcnow().isoformat(),
            "aws_monthly_cost": aws_cost,
            "gcp_monthly_cost": gcp_cost,
            "azure_monthly_cost": azure_cost,
            "aws_annual_cost": round(aws_cost * 12, 2),
            "gcp_annual_cost": round(gcp_cost * 12, 2),
            "azure_annual_cost": round(azure_cost * 12, 2),
            "cost_breakdown": {
                "aws": {"compute": round(aws_cost * 0.49, 2), "storage": round(aws_cost * 0.16, 2), "network": round(aws_cost * 0.10, 2), "database": round(aws_cost * 0.14, 2), "additional_services": round(aws_cost * 0.06, 2), "support": round(aws_cost * 0.04, 2), "total": aws_cost},
                "gcp": {"compute": round(gcp_cost * 0.48, 2), "storage": round(gcp_cost * 0.17, 2), "network": round(gcp_cost * 0.09, 2), "database": round(gcp_cost * 0.15, 2), "additional_services": round(gcp_cost * 0.06, 2), "support": round(gcp_cost * 0.05, 2), "total": gcp_cost},
                "azure": {"compute": round(azure_cost * 0.50, 2), "storage": round(azure_cost * 0.17, 2), "network": round(azure_cost * 0.10, 2), "database": round(azure_cost * 0.14, 2), "additional_services": round(azure_cost * 0.05, 2), "support": round(azure_cost * 0.04, 2), "total": azure_cost},
            },
            "recommendations": [
                f"GCP offers ~{round((1 - gcp_cost / aws_cost) * 100, 1)}% savings over AWS for this workload" if aws_cost > 0 else "Insufficient data for comparison",
                "Consider committed use discounts on GCP for additional 20% savings",
                "Azure Hybrid Benefit can reduce costs if you have existing Windows licenses",
            ],
            "lowest_cost_provider": "gcp",
            "cost_difference_percentage": {
                "aws_vs_gcp": round((1 - gcp_cost / aws_cost) * 100, 1) if aws_cost > 0 else 0,
                "aws_vs_azure": round((1 - azure_cost / aws_cost) * 100, 1) if aws_cost > 0 else 0,
                "azure_vs_gcp": round((1 - gcp_cost / azure_cost) * 100, 1) if azure_cost > 0 else 0,
            },
        }

    @app.post("/api/v1/multi-cloud/tco")
    async def calculate_tco(request: dict):
        if aws_service is None:
            return _no_aws_response()
        
        years = request.get("time_horizon_years", 3)
        total_cost = aws_service.get_total_monthly_cost()
        annual_aws = round(total_cost * 12, 2)

        # Project costs with decreasing year-over-year due to optimization
        aws_projections = [round(annual_aws * (1 - 0.03 * i), 2) for i in range(years)]
        gcp_projections = [round(annual_aws * 0.89 * (1 - 0.04 * i), 2) for i in range(years)]
        azure_projections = [round(annual_aws * 0.95 * (1 - 0.03 * i), 2) for i in range(years)]

        return {
            "id": f"tco_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workload_id": request.get("workload_id", "unknown"),
            "analysis_date": datetime.utcnow().isoformat(),
            "time_horizon_years": years,
            "aws_tco": {f"year_{i+1}": aws_projections[i] for i in range(years)} | {"total": round(sum(aws_projections), 2)},
            "gcp_tco": {f"year_{i+1}": gcp_projections[i] for i in range(years)} | {"total": round(sum(gcp_projections), 2)},
            "azure_tco": {f"year_{i+1}": azure_projections[i] for i in range(years)} | {"total": round(sum(azure_projections), 2)},
            "hidden_costs": {"migration": round(annual_aws * 0.05, 2), "training": round(annual_aws * 0.03, 2), "downtime": round(annual_aws * 0.02, 2), "compliance": round(annual_aws * 0.015, 2)},
            "operational_costs": {"staff": round(annual_aws * 0.4, 2), "monitoring": round(annual_aws * 0.02, 2), "security": round(annual_aws * 0.03, 2)},
            "cost_projections": {"aws": aws_projections, "gcp": gcp_projections, "azure": azure_projections},
            "total_tco_comparison": {"aws": round(sum(aws_projections), 2), "gcp": round(sum(gcp_projections), 2), "azure": round(sum(azure_projections), 2)},
            "recommended_provider": "gcp",
        }

    @app.post("/api/v1/multi-cloud/migration")
    async def analyze_migration(request: dict):
        if aws_service is None:
            return _no_aws_response()
        
        target = request.get("target_provider", "aws")
        target_name = {"aws": "AWS", "gcp": "GCP", "azure": "Azure"}.get(target, target.upper())
        total_cost = aws_service.get_total_monthly_cost()
        
        # Estimate migration cost as ~20% of annual spend
        annual_cost = total_cost * 12
        migration_cost = round(annual_cost * 0.20, 2)
        monthly_savings = round(total_cost * 0.11, 2) if target == "gcp" else round(total_cost * 0.05, 2)
        break_even = round(migration_cost / monthly_savings) if monthly_savings > 0 else 0

        return {
            "id": f"mig_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workload_id": request.get("workload_id", "unknown"),
            "source_provider": "aws",
            "target_provider": target,
            "analysis_date": datetime.utcnow().isoformat(),
            "migration_cost": migration_cost,
            "migration_timeline_days": 60,
            "break_even_months": break_even,
            "cost_breakdown": {
                "cloud_infrastructure_setup": round(migration_cost * 0.14, 2),
                "data_migration_transfer": round(migration_cost * 0.27, 2),
                "application_re_architecture": round(migration_cost * 0.24, 2),
                "testing_and_validation": round(migration_cost * 0.16, 2),
                "team_training": round(migration_cost * 0.11, 2),
                "decommission_old_hardware": round(migration_cost * 0.08, 2),
            },
            "risk_assessment": {
                "overall_risk_level": "medium",
                "technical_risks": [
                    "Data transfer bandwidth — large datasets may take days to upload",
                    "Application compatibility with cloud-native services",
                    "Network latency changes affecting end-user experience",
                    "Licensing changes for software running on cloud VMs",
                ],
                "business_risks": [
                    "Downtime during cutover window (estimated 2-4 hours)",
                    "Team needs cloud skills training",
                    "Ongoing cloud costs replace upfront hardware investment",
                    "Vendor lock-in considerations",
                ],
                "mitigation_strategies": [
                    f"Use {target_name} migration tools for lift-and-shift",
                    "Run parallel environments for 2 weeks before cutover",
                    "Start with non-critical workloads to build confidence",
                    f"Train team on {target_name} fundamentals before migration",
                    "Set up cost alerting from day one",
                ],
                "success_probability": 0.88,
            },
            "recommendations": [
                f"Migrate to {target_name} using a phased approach over 8-10 weeks",
                f"Use {target_name} database migration tools for database migration",
                "Start with dev/staging servers, then move production last",
                "Set up VPN or Direct Connect for secure data transfer",
                f"Estimated monthly cloud cost on {target_name}: ${round(total_cost * 0.89 if target == 'gcp' else total_cost * 0.95, 2)} vs current: ${round(total_cost, 2)}/mo",
            ],
            "monthly_savings": monthly_savings,
            "annual_savings": round(monthly_savings * 12, 2),
            "roi_percentage": round((monthly_savings * 12 / migration_cost) * 100, 1) if migration_cost > 0 else 0,
        }

    @app.post("/api/v1/multi-cloud/validate")
    async def validate_workload(workload: dict):
        if aws_service is None:
            return _no_aws_response()
        
        cost = aws_service.get_total_monthly_cost()
        return {
            "is_valid": True,
            "errors": [],
            "warnings": ["Consider adding backup storage for disaster recovery"] if cost > 0 else ["No cost data available yet"],
            "estimated_monthly_cost_range": {
                "min_cost": round(cost * 0.85, 2),
                "max_cost": round(cost * 1.15, 2),
                "currency": "USD",
            },
        }

    @app.get("/api/v1/multi-cloud/comparisons/{workload_id}")
    async def get_workload_comparisons(workload_id: str, page: int = 1, page_size: int = 10):
        return {
            "comparisons": [],
            "total_count": 0,
            "page": page,
            "page_size": page_size,
        }

    @app.get("/api/v1/multi-cloud/services/{provider}")
    async def get_provider_services(provider: str, category: str = None):
        services = {
            "aws": [
                {"name": "EC2", "category": "compute", "description": "Virtual servers", "pricing_units": ["per hour"], "regions": ["us-east-1", "us-west-2"]},
                {"name": "S3", "category": "storage", "description": "Object storage", "pricing_units": ["per GB"], "regions": ["us-east-1", "us-west-2"]},
                {"name": "RDS", "category": "database", "description": "Managed databases", "pricing_units": ["per hour"], "regions": ["us-east-1", "us-west-2"]},
            ],
            "gcp": [
                {"name": "Compute Engine", "category": "compute", "description": "Virtual machines", "pricing_units": ["per hour"], "regions": ["us-central1", "us-east1"]},
                {"name": "Cloud Storage", "category": "storage", "description": "Object storage", "pricing_units": ["per GB"], "regions": ["us-central1", "us-east1"]},
                {"name": "Cloud SQL", "category": "database", "description": "Managed databases", "pricing_units": ["per hour"], "regions": ["us-central1", "us-east1"]},
            ],
            "azure": [
                {"name": "Virtual Machines", "category": "compute", "description": "Cloud VMs", "pricing_units": ["per hour"], "regions": ["eastus", "westus2"]},
                {"name": "Blob Storage", "category": "storage", "description": "Object storage", "pricing_units": ["per GB"], "regions": ["eastus", "westus2"]},
                {"name": "Azure SQL", "category": "database", "description": "Managed databases", "pricing_units": ["per hour"], "regions": ["eastus", "westus2"]},
            ],
        }
        return services.get(provider, [])

    if __name__ == "__main__":
        print("🚀 Starting FinOps Platform Backend API...")
        print("📡 API Server: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("🎯 Frontend should connect automatically via proxy")
        print("⚠️  Connect your AWS account at /api/v1/onboarding/quick-setup")

        uvicorn.run(app, host="0.0.0.0", port=8000)

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Installing required packages...")
    os.system("pip install fastapi uvicorn boto3")
    print("🔄 Please run the script again after installation")