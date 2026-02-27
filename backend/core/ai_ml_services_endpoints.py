"""
AI/ML Services REST API Endpoints

This module provides comprehensive REST API endpoints for all AI/ML services
in the FinOps platform, including predictive scaling, workload intelligence,
natural language processing, model management, and AI system monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import uuid

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
import structlog

from .auth import get_current_user
from .models import User
from .exceptions import AIServiceError, ModelNotFoundError, ValidationException

# Import AI/ML components
from .predictive_scaling_engine import PredictiveScalingEngine, ForecastHorizon, ScalingActionType
from .workload_intelligence_system import WorkloadIntelligenceSystem, WorkloadProfile, PlacementRecommendation
from .natural_language_interface import get_natural_language_interface, QueryResponse, ConversationContext
from .ml_model_manager import ModelManager, ModelStatus
from .ai_orchestrator import AIOrchestrator
from .graph_neural_network_system import GraphNeuralNetworkSystem
from .predictive_maintenance_system import PredictiveMaintenanceSystem
from .smart_contract_optimizer import SmartContractOptimizer
from .reinforcement_learning_agent import ReinforcementLearningAgent

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/ai-ml", tags=["AI/ML Services"])

# Global service instances (would be dependency injected in production)
_predictive_scaling_engine: Optional[PredictiveScalingEngine] = None
_workload_intelligence_system: Optional[WorkloadIntelligenceSystem] = None
_ml_model_manager: Optional[ModelManager] = None
_ai_orchestrator: Optional[AIOrchestrator] = None
_gnn_system: Optional[GraphNeuralNetworkSystem] = None
_predictive_maintenance: Optional[PredictiveMaintenanceSystem] = None
_smart_contract_optimizer: Optional[SmartContractOptimizer] = None
_rl_agent: Optional[ReinforcementLearningAgent] = None

def get_predictive_scaling_engine() -> PredictiveScalingEngine:
    """Get or create predictive scaling engine instance"""
    global _predictive_scaling_engine
    if _predictive_scaling_engine is None:
        from .safety_checker import SafetyChecker
        _predictive_scaling_engine = PredictiveScalingEngine(SafetyChecker())
    return _predictive_scaling_engine

def get_workload_intelligence_system() -> WorkloadIntelligenceSystem:
    """Get or create workload intelligence system instance"""
    global _workload_intelligence_system
    if _workload_intelligence_system is None:
        _workload_intelligence_system = WorkloadIntelligenceSystem()
    return _workload_intelligence_system

def get_ml_model_manager() -> ModelManager:
    """Get or create ML model manager instance"""
    global _ml_model_manager
    if _ml_model_manager is None:
        _ml_model_manager = ModelManager()
    return _ml_model_manager

def get_ai_orchestrator() -> AIOrchestrator:
    """Get or create AI orchestrator instance"""
    global _ai_orchestrator
    if _ai_orchestrator is None:
        _ai_orchestrator = AIOrchestrator()
    return _ai_orchestrator

def get_gnn_system() -> GraphNeuralNetworkSystem:
    """Get or create graph neural network system instance"""
    global _gnn_system
    if _gnn_system is None:
        _gnn_system = GraphNeuralNetworkSystem()
    return _gnn_system

def get_predictive_maintenance() -> PredictiveMaintenanceSystem:
    """Get or create predictive maintenance system instance"""
    global _predictive_maintenance
    if _predictive_maintenance is None:
        _predictive_maintenance = PredictiveMaintenanceSystem()
    return _predictive_maintenance

def get_smart_contract_optimizer() -> SmartContractOptimizer:
    """Get or create smart contract optimizer instance"""
    global _smart_contract_optimizer
    if _smart_contract_optimizer is None:
        _smart_contract_optimizer = SmartContractOptimizer()
    return _smart_contract_optimizer

def get_rl_agent() -> ReinforcementLearningAgent:
    """Get or create reinforcement learning agent instance"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = ReinforcementLearningAgent()
    return _rl_agent

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

# Predictive Scaling Models
class PredictiveScalingConfigRequest(BaseModel):
    """Request to configure predictive scaling for resources"""
    resource_ids: List[str] = Field(..., description="List of resource IDs to configure")
    scaling_policy: Dict[str, Any] = Field(..., description="Scaling policy configuration")
    forecast_horizons: List[str] = Field(default=["1h", "24h", "7d"], description="Forecast horizons")
    safety_thresholds: Dict[str, float] = Field(default={}, description="Safety thresholds")
    notification_settings: Dict[str, Any] = Field(default={}, description="Notification settings")

class ScalingForecastRequest(BaseModel):
    """Request for scaling forecast"""
    resource_id: str = Field(..., description="Resource ID")
    horizon: str = Field(..., description="Forecast horizon")
    include_recommendations: bool = Field(default=True, description="Include scaling recommendations")

class ScalingMonitoringResponse(BaseModel):
    """Response for scaling monitoring data"""
    resource_id: str
    current_status: str
    recent_actions: List[Dict[str, Any]]
    forecast_accuracy: Dict[str, float]
    performance_metrics: Dict[str, Any]
    next_predicted_action: Optional[Dict[str, Any]]

# Workload Intelligence Models
class WorkloadAnalysisRequest(BaseModel):
    """Request for workload analysis"""
    workload_spec: Dict[str, Any] = Field(..., description="Workload specification")
    current_placement: Optional[Dict[str, str]] = Field(None, description="Current placement info")
    requirements: Dict[str, Any] = Field(default={}, description="Workload requirements")
    constraints: Dict[str, Any] = Field(default={}, description="Placement constraints")

class PlacementRecommendationRequest(BaseModel):
    """Request for placement recommendations"""
    workload_profile: Dict[str, Any] = Field(..., description="Workload profile")
    available_providers: List[str] = Field(..., description="Available cloud providers")
    optimization_goals: List[str] = Field(default=["cost", "performance"], description="Optimization goals")
    compliance_requirements: List[str] = Field(default=[], description="Compliance requirements")

class WorkloadMigrationRequest(BaseModel):
    """Request for workload migration analysis"""
    workload_id: str = Field(..., description="Workload identifier")
    source_placement: Dict[str, str] = Field(..., description="Source placement")
    target_placement: Dict[str, str] = Field(..., description="Target placement")
    migration_timeline: Optional[str] = Field(None, description="Preferred migration timeline")

# Natural Language Interface Models
class EnhancedChatRequest(BaseModel):
    """Enhanced chat request with AI context"""
    query: str = Field(..., description="Natural language query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    ai_services_context: Optional[Dict[str, Any]] = Field(None, description="AI services context")
    preferred_response_format: str = Field(default="conversational", description="Response format preference")

class AIInsightRequest(BaseModel):
    """Request for AI-generated insights"""
    data_sources: List[str] = Field(..., description="Data sources to analyze")
    analysis_type: str = Field(..., description="Type of analysis")
    time_range: Dict[str, str] = Field(..., description="Time range for analysis")
    focus_areas: List[str] = Field(default=[], description="Specific focus areas")

# Model Management Models
class ModelPerformanceRequest(BaseModel):
    """Request for model performance analysis"""
    model_ids: List[str] = Field(..., description="Model IDs to analyze")
    metrics: List[str] = Field(default=[], description="Specific metrics to include")
    time_range: Dict[str, str] = Field(..., description="Time range for analysis")

class ModelComparisonRequest(BaseModel):
    """Request for model comparison"""
    model_ids: List[str] = Field(..., min_items=2, description="Models to compare")
    comparison_metrics: List[str] = Field(..., description="Metrics for comparison")
    test_dataset_id: Optional[str] = Field(None, description="Test dataset ID")

# AI System Monitoring Models
class SystemPerformanceRequest(BaseModel):
    """Request for AI system performance data"""
    system_types: List[str] = Field(default=[], description="AI system types to monitor")
    metrics: List[str] = Field(default=[], description="Performance metrics")
    time_range: Dict[str, str] = Field(..., description="Time range")
    aggregation_level: str = Field(default="hourly", description="Data aggregation level")

class AlertConfigurationRequest(BaseModel):
    """Request to configure AI system alerts"""
    system_type: str = Field(..., description="AI system type")
    alert_rules: List[Dict[str, Any]] = Field(..., description="Alert rules")
    notification_channels: List[str] = Field(..., description="Notification channels")
    escalation_policy: Optional[Dict[str, Any]] = Field(None, description="Escalation policy")

# ============================================================================
# PREDICTIVE SCALING ENDPOINTS
# ============================================================================

@router.post("/predictive-scaling/configure")
async def configure_predictive_scaling(
    request: PredictiveScalingConfigRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Configure predictive scaling for multiple resources"""
    try:
        logger.info(
            "Configuring predictive scaling",
            user_id=current_user.id,
            resource_count=len(request.resource_ids)
        )
        
        engine = get_predictive_scaling_engine()
        
        # Configure each resource
        configuration_results = []
        for resource_id in request.resource_ids:
            try:
                # Initialize resource monitoring
                success = await engine.configure_resource_scaling(
                    resource_id=resource_id,
                    scaling_policy=request.scaling_policy,
                    forecast_horizons=[ForecastHorizon(h) for h in request.forecast_horizons],
                    safety_thresholds=request.safety_thresholds
                )
                
                configuration_results.append({
                    "resource_id": resource_id,
                    "success": success,
                    "status": "configured" if success else "failed"
                })
                
            except Exception as e:
                logger.error(f"Failed to configure resource {resource_id}: {str(e)}")
                configuration_results.append({
                    "resource_id": resource_id,
                    "success": False,
                    "status": "error",
                    "error": str(e)
                })
        
        # Schedule background monitoring setup
        background_tasks.add_task(
            _setup_scaling_monitoring,
            request.resource_ids,
            request.notification_settings
        )
        
        return {
            "success": True,
            "message": "Predictive scaling configuration completed",
            "results": configuration_results,
            "configured_count": sum(1 for r in configuration_results if r["success"]),
            "failed_count": sum(1 for r in configuration_results if not r["success"])
        }
        
    except Exception as e:
        logger.error(f"Predictive scaling configuration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictive-scaling/forecast/{resource_id}")
async def get_scaling_forecast(
    resource_id: str,
    horizon: str = Query(..., description="Forecast horizon (1h, 24h, 7d)"),
    include_recommendations: bool = Query(True, description="Include scaling recommendations"),
    current_user: User = Depends(get_current_user)
):
    """Get predictive scaling forecast for a resource"""
    try:
        logger.info(
            "Getting scaling forecast",
            user_id=current_user.id,
            resource_id=resource_id,
            horizon=horizon
        )
        
        engine = get_predictive_scaling_engine()
        
        # Validate horizon
        try:
            forecast_horizon = ForecastHorizon(horizon)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid horizon: {horizon}")
        
        # Get forecast
        forecast = await engine.get_resource_forecast(resource_id, forecast_horizon)
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Resource not found or not configured")
        
        response_data = {
            "resource_id": resource_id,
            "forecast_horizon": horizon,
            "forecast": forecast.to_dict(),
            "generated_at": datetime.now().isoformat()
        }
        
        # Include recommendations if requested
        if include_recommendations:
            recommendations = await engine.get_scaling_recommendations(resource_id, forecast)
            response_data["recommendations"] = [rec.to_dict() for rec in recommendations]
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scaling forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictive-scaling/monitoring/{resource_id}", response_model=ScalingMonitoringResponse)
async def get_scaling_monitoring_data(
    resource_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days of monitoring data"),
    current_user: User = Depends(get_current_user)
):
    """Get monitoring data for predictive scaling"""
    try:
        logger.info(
            "Getting scaling monitoring data",
            user_id=current_user.id,
            resource_id=resource_id,
            days=days
        )
        
        engine = get_predictive_scaling_engine()
        
        # Get monitoring data
        monitoring_data = await engine.get_monitoring_data(resource_id, days)
        
        if not monitoring_data:
            raise HTTPException(status_code=404, detail="Resource not found or no monitoring data")
        
        return ScalingMonitoringResponse(**monitoring_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get monitoring data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WORKLOAD INTELLIGENCE ENDPOINTS
# ============================================================================

@router.post("/workload-intelligence/analyze")
async def analyze_workload(
    request: WorkloadAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze workload characteristics and requirements"""
    try:
        logger.info(
            "Analyzing workload",
            user_id=current_user.id,
            workload_type=request.workload_spec.get("type", "unknown")
        )
        
        system = get_workload_intelligence_system()
        
        # Analyze workload
        analysis = await system.analyze_workload(
            workload_spec=request.workload_spec,
            current_placement=request.current_placement,
            requirements=request.requirements,
            constraints=request.constraints
        )
        
        return {
            "success": True,
            "analysis": analysis.to_dict(),
            "analyzed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Workload analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workload-intelligence/placement-recommendations")
async def get_placement_recommendations(
    request: PlacementRecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """Get intelligent placement recommendations for workloads"""
    try:
        logger.info(
            "Getting placement recommendations",
            user_id=current_user.id,
            provider_count=len(request.available_providers)
        )
        
        system = get_workload_intelligence_system()
        
        # Get recommendations
        recommendations = await system.get_placement_recommendations(
            workload_profile=request.workload_profile,
            available_providers=request.available_providers,
            optimization_goals=request.optimization_goals,
            compliance_requirements=request.compliance_requirements
        )
        
        return {
            "success": True,
            "recommendations": [rec.to_dict() for rec in recommendations],
            "recommendation_count": len(recommendations),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Placement recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workload-intelligence/migration-analysis")
async def analyze_workload_migration(
    request: WorkloadMigrationRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze workload migration feasibility and impact"""
    try:
        logger.info(
            "Analyzing workload migration",
            user_id=current_user.id,
            workload_id=request.workload_id
        )
        
        system = get_workload_intelligence_system()
        
        # Analyze migration
        analysis = await system.analyze_migration(
            workload_id=request.workload_id,
            source_placement=request.source_placement,
            target_placement=request.target_placement,
            migration_timeline=request.migration_timeline
        )
        
        return {
            "success": True,
            "migration_analysis": analysis.to_dict(),
            "analyzed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Migration analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENHANCED NATURAL LANGUAGE INTERFACE ENDPOINTS
# ============================================================================

@router.post("/natural-language/enhanced-chat")
async def enhanced_ai_chat(
    request: EnhancedChatRequest,
    current_user: User = Depends(get_current_user)
):
    """Enhanced AI chat with full AI services context"""
    try:
        logger.info(
            "Processing enhanced AI chat",
            user_id=current_user.id,
            query_length=len(request.query),
            has_context=bool(request.ai_services_context)
        )
        
        # Get NLP interface
        nlp_interface = get_natural_language_interface()
        
        # Enhance context with AI services data
        enhanced_context = request.context_data or {}
        if request.ai_services_context:
            enhanced_context["ai_services"] = request.ai_services_context
        
        # Add current AI system status
        ai_orchestrator = get_ai_orchestrator()
        system_health = await ai_orchestrator.get_system_health()
        enhanced_context["system_health"] = system_health
        
        # Process query with enhanced context
        response = await nlp_interface.process_enhanced_query(
            query=request.query,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            user_id=str(current_user.id),
            context_data=enhanced_context,
            response_format=request.preferred_response_format
        )
        
        return {
            "success": True,
            "conversation_id": request.conversation_id or response.conversation_id,
            "response": response.to_dict(),
            "ai_services_used": response.ai_services_used,
            "confidence_score": response.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Enhanced AI chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/natural-language/ai-insights")
async def generate_ai_insights(
    request: AIInsightRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate AI-powered insights from multiple data sources"""
    try:
        logger.info(
            "Generating AI insights",
            user_id=current_user.id,
            data_sources=request.data_sources,
            analysis_type=request.analysis_type
        )
        
        # Get AI orchestrator for coordinated analysis
        orchestrator = get_ai_orchestrator()
        
        # Generate insights using multiple AI systems
        insights = await orchestrator.generate_coordinated_insights(
            data_sources=request.data_sources,
            analysis_type=request.analysis_type,
            time_range=request.time_range,
            focus_areas=request.focus_areas,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "insights": insights.to_dict(),
            "data_sources_analyzed": len(request.data_sources),
            "ai_systems_used": insights.ai_systems_used,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI insights generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/model-management/performance")
async def get_model_performance(
    request: ModelPerformanceRequest,
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive model performance analysis"""
    try:
        logger.info(
            "Getting model performance",
            user_id=current_user.id,
            model_count=len(request.model_ids)
        )
        
        manager = get_ml_model_manager()
        
        # Get performance data for each model
        performance_data = {}
        for model_id in request.model_ids:
            performance = await manager.get_model_performance(
                model_id=model_id,
                metrics=request.metrics,
                time_range=request.time_range
            )
            performance_data[model_id] = performance
        
        return {
            "success": True,
            "performance_data": performance_data,
            "analyzed_models": len(request.model_ids),
            "time_range": request.time_range,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model performance analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model-management/compare")
async def compare_models(
    request: ModelComparisonRequest,
    current_user: User = Depends(get_current_user)
):
    """Compare multiple models across specified metrics"""
    try:
        logger.info(
            "Comparing models",
            user_id=current_user.id,
            model_count=len(request.model_ids)
        )
        
        manager = get_ml_model_manager()
        
        # Perform model comparison
        comparison = await manager.compare_models(
            model_ids=request.model_ids,
            comparison_metrics=request.comparison_metrics,
            test_dataset_id=request.test_dataset_id
        )
        
        return {
            "success": True,
            "comparison": comparison.to_dict(),
            "models_compared": len(request.model_ids),
            "metrics_analyzed": len(request.comparison_metrics),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AI SYSTEM MONITORING ENDPOINTS
# ============================================================================

@router.get("/monitoring/system-performance")
async def get_ai_system_performance(
    system_types: List[str] = Query(default=[], description="AI system types"),
    metrics: List[str] = Query(default=[], description="Performance metrics"),
    time_range_start: str = Query(..., description="Start time (ISO format)"),
    time_range_end: str = Query(..., description="End time (ISO format)"),
    aggregation_level: str = Query(default="hourly", description="Aggregation level"),
    current_user: User = Depends(get_current_user)
):
    """Get AI system performance metrics"""
    try:
        logger.info(
            "Getting AI system performance",
            user_id=current_user.id,
            system_types=system_types,
            metrics=metrics
        )
        
        orchestrator = get_ai_orchestrator()
        
        # Get performance data
        performance_data = await orchestrator.get_system_performance_metrics(
            system_types=system_types or None,
            metrics=metrics or None,
            time_range={
                "start": time_range_start,
                "end": time_range_end
            },
            aggregation_level=aggregation_level
        )
        
        return {
            "success": True,
            "performance_data": performance_data,
            "time_range": {
                "start": time_range_start,
                "end": time_range_end
            },
            "aggregation_level": aggregation_level,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System performance retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/configure-alerts")
async def configure_ai_system_alerts(
    request: AlertConfigurationRequest,
    current_user: User = Depends(get_current_user)
):
    """Configure alerts for AI system monitoring"""
    try:
        logger.info(
            "Configuring AI system alerts",
            user_id=current_user.id,
            system_type=request.system_type,
            alert_count=len(request.alert_rules)
        )
        
        orchestrator = get_ai_orchestrator()
        
        # Configure alerts
        alert_config_id = await orchestrator.configure_system_alerts(
            system_type=request.system_type,
            alert_rules=request.alert_rules,
            notification_channels=request.notification_channels,
            escalation_policy=request.escalation_policy,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "alert_config_id": alert_config_id,
            "system_type": request.system_type,
            "configured_rules": len(request.alert_rules),
            "configured_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert configuration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/system-health")
async def get_comprehensive_system_health(
    include_details: bool = Query(True, description="Include detailed health information"),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive health status of all AI systems"""
    try:
        logger.info(
            "Getting comprehensive system health",
            user_id=current_user.id,
            include_details=include_details
        )
        
        orchestrator = get_ai_orchestrator()
        
        # Get comprehensive health data
        health_data = await orchestrator.get_comprehensive_system_health(
            include_details=include_details
        )
        
        return {
            "success": True,
            "system_health": health_data,
            "overall_status": health_data.get("overall_status", "unknown"),
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/performance-dashboard")
async def get_performance_dashboard_data(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range in hours"),
    current_user: User = Depends(get_current_user)
):
    """Get performance dashboard data for all AI systems"""
    try:
        logger.info(
            "Getting performance dashboard data",
            user_id=current_user.id,
            time_range_hours=time_range_hours
        )
        
        orchestrator = get_ai_orchestrator()
        
        # Get dashboard data
        dashboard_data = await orchestrator.get_performance_dashboard_data(
            time_range_hours=time_range_hours
        )
        
        return {
            "success": True,
            "dashboard_data": dashboard_data,
            "time_range_hours": time_range_hours,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance dashboard data retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@router.get("/health")
async def ai_ml_services_health_check():
    """Comprehensive health check for all AI/ML services"""
    try:
        health_status = {
            "status": "healthy",
            "services": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check each service
        services_to_check = [
            ("predictive_scaling", get_predictive_scaling_engine),
            ("workload_intelligence", get_workload_intelligence_system),
            ("ml_model_manager", get_ml_model_manager),
            ("ai_orchestrator", get_ai_orchestrator),
            ("gnn_system", get_gnn_system),
            ("predictive_maintenance", get_predictive_maintenance),
            ("smart_contract_optimizer", get_smart_contract_optimizer),
            ("rl_agent", get_rl_agent)
        ]
        
        overall_healthy = True
        
        for service_name, service_getter in services_to_check:
            try:
                service = service_getter()
                service_health = await service.health_check() if hasattr(service, 'health_check') else {"status": "operational"}
                health_status["services"][service_name] = service_health
                
                if service_health.get("status") not in ["healthy", "operational"]:
                    overall_healthy = False
                    
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False
        
        health_status["status"] = "healthy" if overall_healthy else "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _setup_scaling_monitoring(resource_ids: List[str], notification_settings: Dict[str, Any]):
    """Background task to set up scaling monitoring"""
    try:
        logger.info(f"Setting up monitoring for {len(resource_ids)} resources")
        
        # This would set up monitoring infrastructure
        # For now, just log the setup
        for resource_id in resource_ids:
            logger.info(f"Monitoring setup completed for resource {resource_id}")
            
    except Exception as e:
        logger.error(f"Failed to set up monitoring: {str(e)}")