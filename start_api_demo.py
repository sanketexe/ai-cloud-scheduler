#!/usr/bin/env python3
"""
Start a simple version of the FinOps API for demonstration
"""

import sys
import os
import asyncio
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    
    # Create a simple demo API
    app = FastAPI(
        title="FinOps Platform Demo API",
        description="Demonstration of the Automated Cost Optimization Platform",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "FinOps Automated Cost Optimization Platform",
            "version": "1.0.0",
            "status": "Demo Mode",
            "features": [
                "Automated EC2 instance optimization",
                "Storage cost optimization",
                "Network resource cleanup",
                "Multi-account management",
                "Real-time cost tracking",
                "Safety validation system",
                "Compliance and audit logging"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "services": {
                "automation_engine": "operational",
                "safety_checker": "operational",
                "savings_calculator": "operational",
                "multi_account_manager": "operational"
            }
        }
    
    @app.get("/api/v1/automation/actions")
    async def list_automation_actions():
        return {
            "actions": [
                {
                    "id": "stop_unused_instances",
                    "name": "Stop Unused EC2 Instances",
                    "description": "Stop instances with CPU < 5% for 7+ days",
                    "potential_savings": "$500-2000/month",
                    "safety_level": "high"
                },
                {
                    "id": "upgrade_gp2_to_gp3",
                    "name": "Upgrade GP2 to GP3 Volumes",
                    "description": "Upgrade EBS volumes for better performance and cost",
                    "potential_savings": "$50-500/month",
                    "safety_level": "medium"
                },
                {
                    "id": "release_unused_eips",
                    "name": "Release Unused Elastic IPs",
                    "description": "Release unassociated Elastic IP addresses",
                    "potential_savings": "$3.65 per IP/month",
                    "safety_level": "high"
                },
                {
                    "id": "delete_unattached_volumes",
                    "name": "Delete Unattached EBS Volumes",
                    "description": "Remove volumes not attached to any instance",
                    "potential_savings": "$0.10 per GB/month",
                    "safety_level": "medium"
                }
            ]
        }
    
    @app.get("/api/v1/accounts")
    async def list_accounts():
        return {
            "accounts": [
                {
                    "account_id": "123456789012",
                    "account_name": "production-account",
                    "status": "active",
                    "monthly_cost": 15000,
                    "potential_savings": 1200,
                    "optimization_actions": 5
                },
                {
                    "account_id": "123456789013",
                    "account_name": "development-account",
                    "status": "active",
                    "monthly_cost": 8000,
                    "potential_savings": 800,
                    "optimization_actions": 12
                },
                {
                    "account_id": "123456789014",
                    "account_name": "staging-account",
                    "status": "active",
                    "monthly_cost": 4000,
                    "potential_savings": 400,
                    "optimization_actions": 7
                }
            ],
            "summary": {
                "total_accounts": 3,
                "total_monthly_cost": 27000,
                "total_potential_savings": 2400,
                "savings_percentage": 8.9
            }
        }
    
    @app.post("/api/v1/savings/calculate")
    async def calculate_savings(request_data: dict = None):
        return {
            "calculation_id": "calc_123456",
            "timestamp": datetime.utcnow().isoformat(),
            "results": {
                "monthly_savings": 2400,
                "annual_savings": 28800,
                "actions_analyzed": 24,
                "breakdown": {
                    "ec2_optimization": 1800,
                    "storage_optimization": 400,
                    "network_optimization": 200
                },
                "confidence_level": "high",
                "implementation_effort": "low"
            }
        }
    
    @app.get("/api/v1/compliance/audit")
    async def get_audit_trail():
        return {
            "audit_entries": [
                {
                    "timestamp": "2024-12-16T10:30:00Z",
                    "action": "stop_unused_instance",
                    "resource_id": "i-1234567890abcdef0",
                    "account_id": "123456789012",
                    "user": "system",
                    "status": "completed",
                    "savings": 59.90
                },
                {
                    "timestamp": "2024-12-16T11:15:00Z",
                    "action": "upgrade_gp2_to_gp3",
                    "resource_id": "vol-0987654321fedcba0",
                    "account_id": "123456789013",
                    "user": "system",
                    "status": "completed",
                    "savings": 10.00
                }
            ],
            "summary": {
                "total_entries": 156,
                "total_savings": 12450.75,
                "compliance_status": "compliant"
            }
        }
    
    @app.get("/docs")
    async def api_docs():
        return {
            "message": "API Documentation",
            "swagger_ui": "Available at /docs when running with full FastAPI",
            "endpoints": {
                "GET /": "Root endpoint with platform information",
                "GET /health": "Health check endpoint",
                "GET /api/v1/automation/actions": "List available automation actions",
                "GET /api/v1/accounts": "List managed AWS accounts",
                "POST /api/v1/savings/calculate": "Calculate potential savings",
                "GET /api/v1/compliance/audit": "Access audit trail"
            }
        }
    
    if __name__ == "__main__":
        print("🚀 Starting FinOps Platform Demo API...")
        print("📡 API will be available at: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("\n🎯 Try these endpoints:")
        print("   • GET  /api/v1/automation/actions")
        print("   • GET  /api/v1/accounts")
        print("   • POST /api/v1/savings/calculate")
        print("   • GET  /api/v1/compliance/audit")
        print("\n⏹️  Press Ctrl+C to stop the server")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Please install: pip install fastapi uvicorn")
    print("🔄 Or run the full Docker setup: docker-compose up -d")