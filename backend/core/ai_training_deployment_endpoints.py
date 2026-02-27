"""
AI Training and Deployment Pipeline API Endpoints

REST API endpoints for managing AI training pipelines, model deployments,
canary releases, drift detection, and rollback operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
import pandas as pd

from ..ml.ai_training_deployment_pipeline import (
    AITrainingDeploymentPipeline,
    PipelineConfig,
    DeploymentStrategy,
    ModelStatus,
    DriftType,
    CanaryDeployment,
    ModelMetrics,
    DriftDetectionResult
)
from .auth import get_current_user
from .database import get_db_session

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/ai-training", tags=["AI Training & Deployment"])

# Global pipeline instance (in production, this would be managed differently)
_pipeline_instance: Optional[AITrainingDeploymentPipeline] = None

def get_pipeline() -> AITrainingDeploymentPipeline:
    """Get or create pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AITrainingDeploymentPipeline()
    return _pipeline_instance

# Pydantic models for API requests/responses

class TrainingRequest(BaseModel):
    """Request model for training pipeline execution"""
    model_configs: List[Dict[str, Any]] = Field(..., description="List of model configurations to train")
    account_id: str = Field(default="default", description="Account identifier")
    training_data_source: str = Field(default="cost_data", description="Source of training data")
    pipeline_config: Optional[Dict[str, Any]] = Field(None, description="Custom pipeline configuration")

class TrainingResponse(BaseModel):
    """Response model for training pipeline execution"""
    pipeline_run_id: str
    kubeflow_run_id: Optional[str]
    status: str
    trained_models: List[Dict[str, Any]]
    monitoring_setup: Dict[str, Any]
    timestamp: str

class CanaryDeploymentRequest(BaseModel):
    """Request model for canary deployment"""
    model_id: str = Field(..., description="Model ID to deploy")
    traffic_percentage: float = Field(default=10.0, ge=0.0, le=100.0, description="Percentage of traffic for canary")
    duration_hours: int = Field(default=24, ge=1, le=168, description="Duration of canary deployment in hours")
    success_criteria: Optional[Dict[str, float]] = Field(None, description="Custom success criteria")
    rollback_criteria: Optional[Dict[str, float]] = Field(None, description="Custom rollback criteria")

class CanaryDeploymentResponse(BaseModel):
    """Response model for canary deployment"""
    deployment_id: str
    model_id: str
    traffic_percentage: float
    duration_hours: int
    status: str
    start_time: str
    success_criteria: Dict[str, float]
    rollback_criteria: Dict[str, float]

class DriftDetectionRequest(BaseModel):
    """Request model for drift detection"""
    model_id: str = Field(..., description="Model ID to check for drift")
    baseline_days: int = Field(default=30, ge=1, le=365, description="Days of historical data for baseline")
    current_data: Optional[List[Dict[str, Any]]] = Field(None, description="Current data for comparison")

class DriftDetectionResponse(BaseModel):
    """Response model for drift detection"""
    model_id: str
    detection_timestamp: str
    drift_type: str
    drift_score: float
    drift_threshold: float
    is_drift_detected: bool
    affected_features: List[str]
    confidence_level: float
    recommended_action: str
    statistical_test_results: Dict[str, Any]

class RollbackRequest(BaseModel):
    """Request model for model rollback"""
    model_id: str = Field(..., description="Model ID to rollback")
    reason: str = Field(..., description="Reason for rollback")
    target_model_id: Optional[str] = Field(None, description="Specific model to rollback to")

class RollbackResponse(BaseModel):
    """Response model for model rollback"""
    model_id: str
    previous_model_id: str
    reason: str
    timestamp: str
    status: str

class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status"""
    pipeline_id: str
    active_models: int
    canary_deployments: int
    total_models: int
    drift_alerts: int
    mlflow_experiment: str
    kubeflow_host: str
    monitoring_enabled: bool
    auto_rollback_enabled: bool
    last_updated: str

class ModelListResponse(BaseModel):
    """Response model for model list"""
    models: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

# API Endpoints

@router.post("/pipeline/execute", response_model=TrainingResponse)
async def execute_training_pipeline(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Execute AI training pipeline with automated model training, validation, and deployment
    """
    try:
        pipeline = get_pipeline()
        
        # Create custom pipeline config if provided
        if request.pipeline_config:
            config = PipelineConfig(**request.pipeline_config)
            pipeline.config = config
        
        # Generate sample training data (in production, this would come from actual data sources)
        training_data = pd.DataFrame({
            "cost": [100.0 + i for i in range(1000)],
            "usage": [50.0 + i * 0.1 for i in range(1000)],
            "service_type": ["EC2"] * 500 + ["S3"] * 300 + ["RDS"] * 200,
            "anomaly": [0] * 900 + [1] * 100
        })
        
        # Execute pipeline in background
        def execute_pipeline():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                pipeline.execute_training_pipeline(
                    training_data, request.model_configs, request.account_id
                )
            )
        
        background_tasks.add_task(execute_pipeline)
        
        # Return immediate response
        return TrainingResponse(
            pipeline_run_id=pipeline.config.pipeline_id,
            kubeflow_run_id=None,
            status="started",
            trained_models=[],
            monitoring_setup={},
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training pipeline execution failed: {str(e)}")

@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive AI training pipeline status
    """
    try:
        pipeline = get_pipeline()
        status = await pipeline.get_pipeline_status()
        
        return PipelineStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")

@router.post("/deployment/canary", response_model=CanaryDeploymentResponse)
async def deploy_model_canary(
    request: CanaryDeploymentRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Deploy model using canary deployment strategy with traffic splitting
    """
    try:
        pipeline = get_pipeline()
        
        canary = await pipeline.deploy_model_canary(
            model_id=request.model_id,
            traffic_percentage=request.traffic_percentage,
            duration_hours=request.duration_hours
        )
        
        # Update success/rollback criteria if provided
        if request.success_criteria:
            canary.success_criteria.update(request.success_criteria)
        if request.rollback_criteria:
            canary.rollback_criteria.update(request.rollback_criteria)
        
        return CanaryDeploymentResponse(
            deployment_id=canary.deployment_id,
            model_id=canary.model_id,
            traffic_percentage=canary.traffic_percentage,
            duration_hours=canary.duration_hours,
            status=canary.status,
            start_time=canary.start_time.isoformat(),
            success_criteria=canary.success_criteria,
            rollback_criteria=canary.rollback_criteria
        )
        
    except Exception as e:
        logger.error(f"Canary deployment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Canary deployment failed: {str(e)}")

@router.get("/deployment/canary/{deployment_id}")
async def get_canary_deployment_status(
    deployment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get status and metrics for a specific canary deployment
    """
    try:
        pipeline = get_pipeline()
        
        if deployment_id not in pipeline.active_deployments:
            raise HTTPException(status_code=404, detail="Canary deployment not found")
        
        canary = pipeline.active_deployments[deployment_id]
        
        response = {
            "deployment_id": canary.deployment_id,
            "model_id": canary.model_id,
            "status": canary.status,
            "traffic_percentage": canary.traffic_percentage,
            "start_time": canary.start_time.isoformat(),
            "duration_hours": canary.duration_hours,
            "baseline_metrics": canary.baseline_metrics.__dict__ if canary.baseline_metrics else None,
            "canary_metrics": canary.canary_metrics.__dict__ if canary.canary_metrics else None,
            "performance_comparison": canary.performance_comparison,
            "success_criteria": canary.success_criteria,
            "rollback_criteria": canary.rollback_criteria
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get canary deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get canary deployment status: {str(e)}")

@router.post("/drift/detect", response_model=DriftDetectionResponse)
async def detect_model_drift(
    request: DriftDetectionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect model drift using statistical methods and current data
    """
    try:
        pipeline = get_pipeline()
        
        # Convert current data to DataFrame if provided
        if request.current_data:
            current_data = pd.DataFrame(request.current_data)
        else:
            # Generate sample data for demonstration
            current_data = pd.DataFrame({
                "feature_1": [1.0 + i * 0.1 for i in range(100)],
                "feature_2": [2.0 + i * 0.05 for i in range(100)],
                "feature_3": [0.5 + i * 0.02 for i in range(100)]
            })
        
        drift_result = await pipeline.detect_model_drift(
            model_id=request.model_id,
            current_data=current_data,
            baseline_days=request.baseline_days
        )
        
        return DriftDetectionResponse(
            model_id=drift_result.model_id,
            detection_timestamp=drift_result.detection_timestamp.isoformat(),
            drift_type=drift_result.drift_type.value,
            drift_score=drift_result.drift_score,
            drift_threshold=drift_result.drift_threshold,
            is_drift_detected=drift_result.is_drift_detected,
            affected_features=drift_result.affected_features,
            confidence_level=drift_result.confidence_level,
            recommended_action=drift_result.recommended_action,
            statistical_test_results=drift_result.statistical_test_results
        )
        
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

@router.get("/drift/history/{model_id}")
async def get_drift_history(
    model_id: str,
    days: int = Query(default=30, ge=1, le=365, description="Number of days of history to retrieve"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get drift detection history for a specific model
    """
    try:
        pipeline = get_pipeline()
        
        if model_id not in pipeline.drift_history:
            return {"model_id": model_id, "drift_history": [], "total_checks": 0}
        
        # Filter by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        drift_history = [
            {
                "detection_timestamp": result.detection_timestamp.isoformat(),
                "drift_type": result.drift_type.value,
                "drift_score": result.drift_score,
                "is_drift_detected": result.is_drift_detected,
                "affected_features": result.affected_features,
                "confidence_level": result.confidence_level,
                "recommended_action": result.recommended_action
            }
            for result in pipeline.drift_history[model_id]
            if result.detection_timestamp >= cutoff_date
        ]
        
        return {
            "model_id": model_id,
            "drift_history": drift_history,
            "total_checks": len(drift_history),
            "drift_detected_count": len([h for h in drift_history if h["is_drift_detected"]]),
            "date_range": {
                "start": cutoff_date.isoformat(),
                "end": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get drift history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift history: {str(e)}")

@router.post("/rollback", response_model=RollbackResponse)
async def rollback_model(
    request: RollbackRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Rollback model deployment to previous stable version
    """
    try:
        pipeline = get_pipeline()
        
        rollback_result = await pipeline.rollback_model(
            model_id=request.model_id,
            reason=request.reason
        )
        
        return RollbackResponse(**rollback_result)
        
    except Exception as e:
        logger.error(f"Model rollback failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model rollback failed: {str(e)}")

@router.get("/models", response_model=ModelListResponse)
async def list_models(
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    status: Optional[str] = Query(None, description="Filter by model status"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    current_user: dict = Depends(get_current_user)
):
    """
    List models with filtering and pagination
    """
    try:
        pipeline = get_pipeline()
        
        # Get all models
        all_models = []
        for model_id, model_info in pipeline.model_registry.items():
            model_data = {
                "model_id": model_id,
                "model_type": model_info["model_type"],
                "account_id": model_info["account_id"],
                "status": model_info["status"].value,
                "created_at": model_info["created_at"].isoformat(),
                "metrics": model_info["metrics"].__dict__ if hasattr(model_info["metrics"], '__dict__') else model_info["metrics"],
                "mlflow_run_id": model_info.get("mlflow_run_id"),
                "config": model_info["config"]
            }
            all_models.append(model_data)
        
        # Apply filters
        filtered_models = all_models
        
        if account_id:
            filtered_models = [m for m in filtered_models if m["account_id"] == account_id]
        
        if status:
            filtered_models = [m for m in filtered_models if m["status"] == status]
        
        if model_type:
            filtered_models = [m for m in filtered_models if m["model_type"] == model_type]
        
        # Apply pagination
        total_count = len(filtered_models)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_models = filtered_models[start_idx:end_idx]
        
        return ModelListResponse(
            models=paginated_models,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/{model_id}")
async def get_model_details(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information about a specific model
    """
    try:
        pipeline = get_pipeline()
        
        if model_id not in pipeline.model_registry:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = pipeline.model_registry[model_id]
        
        # Get drift history
        drift_history = []
        if model_id in pipeline.drift_history:
            drift_history = [
                {
                    "detection_timestamp": result.detection_timestamp.isoformat(),
                    "drift_type": result.drift_type.value,
                    "drift_score": result.drift_score,
                    "is_drift_detected": result.is_drift_detected
                }
                for result in pipeline.drift_history[model_id][-10:]  # Last 10 checks
            ]
        
        # Get performance history
        performance_history = []
        if model_id in pipeline.performance_history:
            performance_history = [
                metrics.__dict__ if hasattr(metrics, '__dict__') else metrics
                for metrics in pipeline.performance_history[model_id][-10:]  # Last 10 records
            ]
        
        # Get canary deployment info if exists
        canary_deployment = None
        for canary in pipeline.active_deployments.values():
            if canary.model_id == model_id:
                canary_deployment = {
                    "deployment_id": canary.deployment_id,
                    "status": canary.status,
                    "traffic_percentage": canary.traffic_percentage,
                    "start_time": canary.start_time.isoformat()
                }
                break
        
        response = {
            "model_id": model_id,
            "model_type": model_info["model_type"],
            "account_id": model_info["account_id"],
            "status": model_info["status"].value,
            "created_at": model_info["created_at"].isoformat(),
            "metrics": model_info["metrics"].__dict__ if hasattr(model_info["metrics"], '__dict__') else model_info["metrics"],
            "config": model_info["config"],
            "mlflow_run_id": model_info.get("mlflow_run_id"),
            "drift_history": drift_history,
            "performance_history": performance_history,
            "canary_deployment": canary_deployment
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    force: bool = Query(default=False, description="Force delete even if model is active"),
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a model from the registry
    """
    try:
        pipeline = get_pipeline()
        
        if model_id not in pipeline.model_registry:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = pipeline.model_registry[model_id]
        
        # Check if model is active
        if model_info["status"] in [ModelStatus.PRODUCTION, ModelStatus.CANARY] and not force:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete active model. Use force=true to override."
            )
        
        # Remove from registry
        del pipeline.model_registry[model_id]
        
        # Clean up related data
        if model_id in pipeline.drift_history:
            del pipeline.drift_history[model_id]
        
        if model_id in pipeline.performance_history:
            del pipeline.performance_history[model_id]
        
        # Remove canary deployments
        deployments_to_remove = [
            dep_id for dep_id, canary in pipeline.active_deployments.items()
            if canary.model_id == model_id
        ]
        for dep_id in deployments_to_remove:
            del pipeline.active_deployments[dep_id]
        
        return {"message": f"Model {model_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.post("/cleanup")
async def cleanup_old_deployments(
    days_to_keep: int = Query(default=30, ge=1, le=365, description="Days of data to keep"),
    current_user: dict = Depends(get_current_user)
):
    """
    Clean up old model deployments and artifacts
    """
    try:
        pipeline = get_pipeline()
        
        cleanup_result = await pipeline.cleanup_old_deployments(days_to_keep)
        
        return {
            "message": "Cleanup completed successfully",
            "models_cleaned": cleanup_result["models_cleaned"],
            "deployments_cleaned": cleanup_result["deployments_cleaned"],
            "cutoff_date": cleanup_result["cutoff_date"]
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for AI training pipeline
    """
    try:
        pipeline = get_pipeline()
        status = await pipeline.get_pipeline_status()
        
        return {
            "status": "healthy",
            "pipeline_id": status["pipeline_id"],
            "active_models": status["active_models"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Include router in main application
def include_router(app):
    """Include AI training deployment router in FastAPI app"""
    app.include_router(router)