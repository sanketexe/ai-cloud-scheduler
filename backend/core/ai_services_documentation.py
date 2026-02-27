"""
AI/ML Services Documentation and Discovery Endpoint

This module provides comprehensive documentation and discovery capabilities
for all AI/ML services available in the FinOps platform.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, List, Any
from datetime import datetime

from .auth import get_current_user
from .models import User

router = APIRouter(prefix="/api/v1/ai-services", tags=["AI Services Documentation"])

class ServiceEndpoint(BaseModel):
    """Model for service endpoint information"""
    path: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    response_format: Dict[str, Any]

class AIServiceInfo(BaseModel):
    """Model for AI service information"""
    service_name: str
    description: str
    capabilities: List[str]
    endpoints: List[ServiceEndpoint]
    status: str
    version: str

@router.get("/catalog")
async def get_ai_services_catalog(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive catalog of all AI/ML services"""
    
    catalog = {
        "predictive_scaling": {
            "service_name": "Predictive Scaling Engine",
            "description": "AI-powered predictive resource scaling with time series forecasting",
            "capabilities": [
                "Demand forecasting (1h, 24h, 7d horizons)",
                "Automatic scaling recommendations",
                "Safety checks and gradual scaling",
                "Multi-resource optimization",
                "Seasonal pattern detection"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/ai-ml/predictive-scaling/configure",
                    "method": "POST",
                    "description": "Configure predictive scaling for resources",
                    "parameters": ["resource_ids", "scaling_policy", "forecast_horizons"],
                    "response_format": {"success": "boolean", "results": "array"}
                },
                {
                    "path": "/api/v1/ai-ml/predictive-scaling/forecast/{resource_id}",
                    "method": "GET",
                    "description": "Get scaling forecast for a resource",
                    "parameters": ["resource_id", "horizon", "include_recommendations"],
                    "response_format": {"forecast": "object", "recommendations": "array"}
                },
                {
                    "path": "/api/v1/ai-ml/predictive-scaling/monitoring/{resource_id}",
                    "method": "GET",
                    "description": "Get monitoring data for predictive scaling",
                    "parameters": ["resource_id", "days"],
                    "response_format": {"monitoring_data": "object"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        },
        
        "workload_intelligence": {
            "service_name": "Workload Intelligence System",
            "description": "ML-driven optimal workload placement and migration analysis",
            "capabilities": [
                "Workload profiling and analysis",
                "Multi-cloud placement optimization",
                "Performance and cost prediction",
                "Migration feasibility analysis",
                "Compliance-aware recommendations"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/ai-ml/workload-intelligence/analyze",
                    "method": "POST",
                    "description": "Analyze workload characteristics",
                    "parameters": ["workload_spec", "requirements", "constraints"],
                    "response_format": {"analysis": "object"}
                },
                {
                    "path": "/api/v1/ai-ml/workload-intelligence/placement-recommendations",
                    "method": "POST",
                    "description": "Get intelligent placement recommendations",
                    "parameters": ["workload_profile", "available_providers", "optimization_goals"],
                    "response_format": {"recommendations": "array"}
                },
                {
                    "path": "/api/v1/ai-ml/workload-intelligence/migration-analysis",
                    "method": "POST",
                    "description": "Analyze workload migration feasibility",
                    "parameters": ["workload_id", "source_placement", "target_placement"],
                    "response_format": {"migration_analysis": "object"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        },
        
        "natural_language_interface": {
            "service_name": "Natural Language Interface",
            "description": "Conversational AI for cost analysis and optimization queries",
            "capabilities": [
                "Natural language query processing",
                "Context-aware conversations",
                "Automated insight generation",
                "Visualization creation",
                "Multi-turn dialogue support"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/ai-ml/natural-language/enhanced-chat",
                    "method": "POST",
                    "description": "Enhanced AI chat with full context",
                    "parameters": ["query", "conversation_id", "ai_services_context"],
                    "response_format": {"response": "object", "confidence_score": "float"}
                },
                {
                    "path": "/api/v1/ai-ml/natural-language/ai-insights",
                    "method": "POST",
                    "description": "Generate AI-powered insights",
                    "parameters": ["data_sources", "analysis_type", "focus_areas"],
                    "response_format": {"insights": "object", "ai_systems_used": "array"}
                },
                {
                    "path": "/api/v1/nlp/chat",
                    "method": "POST",
                    "description": "Basic chat interface",
                    "parameters": ["query", "conversation_id"],
                    "response_format": {"response": "object"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        },
        
        "reinforcement_learning": {
            "service_name": "Reinforcement Learning Agent",
            "description": "Self-improving optimization through trial and feedback",
            "capabilities": [
                "Adaptive optimization strategies",
                "Continuous learning from outcomes",
                "A/B testing for strategy comparison",
                "Policy performance tracking",
                "Risk-aware decision making"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/rl-optimization/select-action",
                    "method": "POST",
                    "description": "Select optimal optimization action",
                    "parameters": ["system_state", "available_actions", "constraints"],
                    "response_format": {"action": "object", "confidence": "float", "alternatives": "array"}
                },
                {
                    "path": "/api/v1/rl-optimization/feedback",
                    "method": "POST",
                    "description": "Submit action feedback for learning",
                    "parameters": ["action_id", "outcome", "success"],
                    "response_format": {"calculated_reward": "float", "policy_updated": "boolean"}
                },
                {
                    "path": "/api/v1/rl-optimization/policy-performance",
                    "method": "GET",
                    "description": "Get RL policy performance metrics",
                    "parameters": ["days"],
                    "response_format": {"performance_metrics": "object", "success_rate": "float"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        },
        
        "graph_neural_network": {
            "service_name": "Graph Neural Network System",
            "description": "Complex resource relationship analysis and optimization",
            "capabilities": [
                "Resource dependency mapping",
                "Cascade effect prediction",
                "Resource cluster identification",
                "Coordinated optimization",
                "Critical path analysis"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/graph-analysis/build-graph/{account_id}",
                    "method": "POST",
                    "description": "Build resource graph for account",
                    "parameters": ["account_id", "resources"],
                    "response_format": {"nodes": "array", "edges": "array", "metadata": "object"}
                },
                {
                    "path": "/api/v1/graph-analysis/analyze-dependencies",
                    "method": "POST",
                    "description": "Analyze resource dependencies",
                    "parameters": ["graph_data"],
                    "response_format": {"critical_paths": "array", "bottlenecks": "array"}
                },
                {
                    "path": "/api/v1/graph-analysis/predict-cascade-effects",
                    "method": "POST",
                    "description": "Predict cascade effects of actions",
                    "parameters": ["action", "graph_data"],
                    "response_format": {"primary_impact": "object", "secondary_impacts": "array"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        },
        
        "smart_contract_optimizer": {
            "service_name": "Smart Contract Optimizer",
            "description": "AI-optimized reserved instance and savings plan recommendations",
            "capabilities": [
                "Usage pattern analysis",
                "RI recommendation with confidence intervals",
                "Multi-cloud commitment optimization",
                "Market condition analysis",
                "Portfolio rebalancing"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/smart-contracts/analyze-usage",
                    "method": "POST",
                    "description": "Analyze usage patterns for commitments",
                    "parameters": ["account_id", "resource_types", "analysis_period_days"],
                    "response_format": {"usage_patterns": "array"}
                },
                {
                    "path": "/api/v1/smart-contracts/ri-recommendations",
                    "method": "POST",
                    "description": "Get RI recommendations",
                    "parameters": ["usage_patterns", "risk_tolerance"],
                    "response_format": {"recommendations": "array"}
                },
                {
                    "path": "/api/v1/smart-contracts/optimize-commitments",
                    "method": "POST",
                    "description": "Optimize commitment portfolio",
                    "parameters": ["current_commitments", "usage_forecasts"],
                    "response_format": {"portfolio": "object", "recommendations": "array"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        },
        
        "ml_model_management": {
            "service_name": "ML Model Management Platform",
            "description": "Comprehensive ML model lifecycle and experimentation",
            "capabilities": [
                "Model training and validation",
                "A/B testing framework",
                "Experiment tracking",
                "Model interpretation",
                "Bias detection and mitigation"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/ml/models/train",
                    "method": "POST",
                    "description": "Train new ML model",
                    "parameters": ["model_config", "training_data"],
                    "response_format": {"model_id": "string", "training_results": "object"}
                },
                {
                    "path": "/api/v1/ml/ab-tests",
                    "method": "POST",
                    "description": "Create A/B test",
                    "parameters": ["test_config", "variants"],
                    "response_format": {"test_id": "string"}
                },
                {
                    "path": "/api/v1/ai-ml/model-management/performance",
                    "method": "GET",
                    "description": "Get model performance analysis",
                    "parameters": ["model_ids", "metrics", "time_range"],
                    "response_format": {"performance_data": "object"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        },
        
        "ai_orchestrator": {
            "service_name": "AI Orchestrator",
            "description": "Coordinated AI system management and optimization",
            "capabilities": [
                "Multi-system coordination",
                "Intelligent resource coordination",
                "Contextual recommendations",
                "System health monitoring",
                "User preference learning"
            ],
            "endpoints": [
                {
                    "path": "/api/v1/ai-orchestrator/optimize",
                    "method": "POST",
                    "description": "Coordinate optimization across AI systems",
                    "parameters": ["optimization_context", "goals", "constraints"],
                    "response_format": {"coordinated_recommendation": "object"}
                },
                {
                    "path": "/api/v1/ai-orchestrator/health",
                    "method": "GET",
                    "description": "Get AI system health status",
                    "parameters": [],
                    "response_format": {"systems": "object", "orchestrator": "object"}
                },
                {
                    "path": "/api/v1/ai-ml/monitoring/system-performance",
                    "method": "GET",
                    "description": "Get AI system performance metrics",
                    "parameters": ["system_types", "metrics", "time_range"],
                    "response_format": {"performance_data": "object"}
                }
            ],
            "status": "operational",
            "version": "1.0.0"
        }
    }
    
    return {
        "success": True,
        "ai_services_catalog": catalog,
        "total_services": len(catalog),
        "generated_at": datetime.now().isoformat()
    }

@router.get("/capabilities-matrix")
async def get_capabilities_matrix(
    current_user: User = Depends(get_current_user)
):
    """Get matrix of AI/ML capabilities across services"""
    
    capabilities_matrix = {
        "forecasting": {
            "services": ["predictive_scaling", "smart_contract_optimizer"],
            "description": "Time series forecasting and demand prediction"
        },
        "optimization": {
            "services": ["reinforcement_learning", "workload_intelligence", "smart_contract_optimizer", "ai_orchestrator"],
            "description": "Resource and cost optimization"
        },
        "natural_language": {
            "services": ["natural_language_interface"],
            "description": "Natural language processing and conversation"
        },
        "machine_learning": {
            "services": ["ml_model_management", "reinforcement_learning"],
            "description": "ML model training, validation, and management"
        },
        "graph_analysis": {
            "services": ["graph_neural_network"],
            "description": "Complex relationship and dependency analysis"
        },
        "real_time_processing": {
            "services": ["predictive_scaling", "ai_orchestrator"],
            "description": "Real-time data processing and decision making"
        },
        "multi_cloud": {
            "services": ["workload_intelligence", "smart_contract_optimizer"],
            "description": "Multi-cloud provider optimization"
        },
        "continuous_learning": {
            "services": ["reinforcement_learning", "ml_model_management"],
            "description": "Adaptive learning and improvement"
        },
        "risk_assessment": {
            "services": ["smart_contract_optimizer", "reinforcement_learning", "graph_neural_network"],
            "description": "Risk analysis and mitigation"
        },
        "visualization": {
            "services": ["natural_language_interface", "ai_orchestrator"],
            "description": "Data visualization and reporting"
        }
    }
    
    return {
        "success": True,
        "capabilities_matrix": capabilities_matrix,
        "total_capabilities": len(capabilities_matrix),
        "generated_at": datetime.now().isoformat()
    }

@router.get("/integration-guide")
async def get_integration_guide(
    current_user: User = Depends(get_current_user)
):
    """Get integration guide for AI/ML services"""
    
    integration_guide = {
        "getting_started": {
            "step_1": {
                "title": "Authentication",
                "description": "Obtain API authentication token",
                "endpoint": "/api/v1/auth/login",
                "required": True
            },
            "step_2": {
                "title": "Health Check",
                "description": "Verify AI services are operational",
                "endpoint": "/api/v1/ai-ml/health",
                "required": True
            },
            "step_3": {
                "title": "Service Selection",
                "description": "Choose appropriate AI services for your use case",
                "endpoint": "/api/v1/ai-services/catalog",
                "required": True
            }
        },
        
        "common_workflows": {
            "cost_optimization": {
                "description": "End-to-end cost optimization workflow",
                "steps": [
                    "Analyze usage patterns with Smart Contract Optimizer",
                    "Get workload placement recommendations",
                    "Configure predictive scaling",
                    "Monitor with AI Orchestrator"
                ],
                "estimated_time": "30-60 minutes"
            },
            "intelligent_scaling": {
                "description": "Set up intelligent resource scaling",
                "steps": [
                    "Configure predictive scaling for resources",
                    "Set up monitoring and alerts",
                    "Enable reinforcement learning feedback",
                    "Monitor performance and adjust"
                ],
                "estimated_time": "15-30 minutes"
            },
            "workload_migration": {
                "description": "AI-assisted workload migration",
                "steps": [
                    "Analyze current workload with Workload Intelligence",
                    "Get placement recommendations",
                    "Analyze migration feasibility",
                    "Execute migration with monitoring"
                ],
                "estimated_time": "45-90 minutes"
            }
        },
        
        "best_practices": [
            "Always check service health before making requests",
            "Use appropriate time ranges for historical data analysis",
            "Implement proper error handling for AI service calls",
            "Monitor AI system performance regularly",
            "Provide feedback to reinforcement learning systems",
            "Use conversation IDs for multi-turn natural language interactions",
            "Configure alerts for AI system anomalies",
            "Regularly review and update AI service configurations"
        ],
        
        "rate_limits": {
            "predictive_scaling": "100 requests/minute",
            "workload_intelligence": "50 requests/minute",
            "natural_language": "200 requests/minute",
            "reinforcement_learning": "100 requests/minute",
            "graph_analysis": "25 requests/minute",
            "smart_contracts": "50 requests/minute",
            "model_management": "25 requests/minute",
            "ai_orchestrator": "100 requests/minute"
        },
        
        "error_codes": {
            "400": "Bad Request - Invalid parameters or request format",
            "401": "Unauthorized - Invalid or missing authentication",
            "403": "Forbidden - Insufficient permissions",
            "404": "Not Found - Resource or service not found",
            "422": "Unprocessable Entity - AI processing error",
            "429": "Too Many Requests - Rate limit exceeded",
            "500": "Internal Server Error - AI service error",
            "503": "Service Unavailable - AI service temporarily unavailable"
        }
    }
    
    return {
        "success": True,
        "integration_guide": integration_guide,
        "generated_at": datetime.now().isoformat()
    }

@router.get("/health-dashboard")
async def get_ai_services_health_dashboard(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive health dashboard for all AI services"""
    
    # This would typically call the actual health check endpoints
    # For now, return a mock dashboard
    health_dashboard = {
        "overall_status": "healthy",
        "services_status": {
            "predictive_scaling": {"status": "healthy", "response_time_ms": 45, "uptime": "99.9%"},
            "workload_intelligence": {"status": "healthy", "response_time_ms": 67, "uptime": "99.8%"},
            "natural_language": {"status": "healthy", "response_time_ms": 123, "uptime": "99.7%"},
            "reinforcement_learning": {"status": "healthy", "response_time_ms": 89, "uptime": "99.9%"},
            "graph_neural_network": {"status": "healthy", "response_time_ms": 156, "uptime": "99.6%"},
            "smart_contract_optimizer": {"status": "healthy", "response_time_ms": 78, "uptime": "99.8%"},
            "ml_model_management": {"status": "healthy", "response_time_ms": 234, "uptime": "99.5%"},
            "ai_orchestrator": {"status": "healthy", "response_time_ms": 56, "uptime": "99.9%"}
        },
        "performance_metrics": {
            "total_requests_24h": 15420,
            "average_response_time_ms": 92,
            "error_rate_percent": 0.2,
            "successful_requests_percent": 99.8
        },
        "resource_usage": {
            "cpu_usage_percent": 45,
            "memory_usage_percent": 62,
            "gpu_usage_percent": 78,
            "storage_usage_percent": 34
        },
        "alerts": [
            {
                "service": "graph_neural_network",
                "level": "warning",
                "message": "Response time above threshold",
                "timestamp": "2024-01-04T10:30:00Z"
            }
        ]
    }
    
    return {
        "success": True,
        "health_dashboard": health_dashboard,
        "last_updated": datetime.now().isoformat()
    }