"""
Task Management API Endpoints
Provides REST API for managing background tasks
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
import structlog

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field

from .auth import get_current_user, require_permission
from .models import User, UserRole
from .tasks import (
    TaskMonitor, trigger_cost_sync, trigger_resource_discovery,
    trigger_budget_monitoring, trigger_optimization_analysis,
    sync_provider_cost_data, sync_all_providers_cost_data,
    discover_provider_resources, discover_all_resources,
    monitor_budget, monitor_all_budgets, analyze_optimization_opportunities,
    detect_cost_anomalies, cleanup_old_data
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["Task Management"])

# Request/Response Models
class TaskTriggerRequest(BaseModel):
    """Request to trigger a task"""
    provider_id: Optional[UUID] = Field(None, description="Provider ID for provider-specific tasks")
    budget_id: Optional[UUID] = Field(None, description="Budget ID for budget-specific tasks")
    start_date: Optional[date] = Field(None, description="Start date for date-range tasks")
    end_date: Optional[date] = Field(None, description="End date for date-range tasks")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional task parameters")

class TaskResponse(BaseModel):
    """Task execution response"""
    task_id: str = Field(..., description="Unique task identifier")
    task_name: str = Field(..., description="Name of the triggered task")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Status message")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")

class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Current task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
    traceback: Optional[str] = Field(None, description="Error traceback if failed")
    date_done: Optional[datetime] = Field(None, description="Completion timestamp")
    progress: Optional[Dict[str, Any]] = Field(None, description="Task progress information")

class ActiveTaskResponse(BaseModel):
    """Active task information"""
    worker: str = Field(..., description="Worker executing the task")
    task_id: str = Field(..., description="Task identifier")
    name: str = Field(..., description="Task name")
    args: List[Any] = Field(default_factory=list, description="Task arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Task keyword arguments")
    time_start: Optional[float] = Field(None, description="Task start timestamp")

class WorkerStatsResponse(BaseModel):
    """Worker statistics"""
    workers: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Worker statistics by name")
    total_workers: int = Field(..., description="Total number of active workers")

# Cost Data Sync Endpoints
@router.post("/sync/cost-data", response_model=TaskResponse)
async def trigger_cost_data_sync(
    request: TaskTriggerRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger cost data synchronization for a specific provider or all providers
    
    Requires: Finance Manager or Admin role
    """
    require_permission(current_user.role, "tasks", "create")
    
    try:
        if request.provider_id:
            # Sync specific provider
            task_id = trigger_cost_sync(
                request.provider_id,
                request.start_date,
                request.end_date
            )
            task_name = f"sync_provider_cost_data"
            message = f"Cost data sync triggered for provider {request.provider_id}"
        else:
            # Sync all providers
            task = sync_all_providers_cost_data.delay()
            task_id = task.id
            task_name = "sync_all_providers_cost_data"
            message = "Cost data sync triggered for all providers"
        
        logger.info("Cost data sync triggered",
                   task_id=task_id,
                   provider_id=request.provider_id,
                   user_id=current_user.id)
        
        return TaskResponse(
            task_id=task_id,
            task_name=task_name,
            status="PENDING",
            message=message,
            estimated_duration=300  # 5 minutes
        )
        
    except Exception as e:
        logger.error("Failed to trigger cost data sync",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to trigger task: {str(e)}")

@router.post("/sync/resources", response_model=TaskResponse)
async def trigger_resource_discovery_sync(
    request: TaskTriggerRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger resource discovery for a specific provider or all providers
    
    Requires: Finance Manager or Admin role
    """
    require_permission(current_user.role, "tasks", "create")
    
    try:
        if request.provider_id:
            # Discover resources for specific provider
            task_id = trigger_resource_discovery(request.provider_id)
            task_name = "discover_provider_resources"
            message = f"Resource discovery triggered for provider {request.provider_id}"
        else:
            # Discover resources for all providers
            task = discover_all_resources.delay()
            task_id = task.id
            task_name = "discover_all_resources"
            message = "Resource discovery triggered for all providers"
        
        logger.info("Resource discovery triggered",
                   task_id=task_id,
                   provider_id=request.provider_id,
                   user_id=current_user.id)
        
        return TaskResponse(
            task_id=task_id,
            task_name=task_name,
            status="PENDING",
            message=message,
            estimated_duration=180  # 3 minutes
        )
        
    except Exception as e:
        logger.error("Failed to trigger resource discovery",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to trigger task: {str(e)}")

# Budget Monitoring Endpoints
@router.post("/monitor/budgets", response_model=TaskResponse)
async def trigger_budget_monitoring_task(
    request: TaskTriggerRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger budget monitoring for a specific budget or all budgets
    
    Requires: Finance Manager or Admin role
    """
    require_permission(current_user.role, "budgets", "read")
    
    try:
        if request.budget_id:
            # Monitor specific budget
            task_id = trigger_budget_monitoring(request.budget_id)
            task_name = "monitor_budget"
            message = f"Budget monitoring triggered for budget {request.budget_id}"
        else:
            # Monitor all budgets
            task = monitor_all_budgets.delay()
            task_id = task.id
            task_name = "monitor_all_budgets"
            message = "Budget monitoring triggered for all budgets"
        
        logger.info("Budget monitoring triggered",
                   task_id=task_id,
                   budget_id=request.budget_id,
                   user_id=current_user.id)
        
        return TaskResponse(
            task_id=task_id,
            task_name=task_name,
            status="PENDING",
            message=message,
            estimated_duration=60  # 1 minute
        )
        
    except Exception as e:
        logger.error("Failed to trigger budget monitoring",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to trigger task: {str(e)}")

# Analysis Endpoints
@router.post("/analyze/optimization", response_model=TaskResponse)
async def trigger_optimization_analysis_task(
    request: TaskTriggerRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger optimization analysis for a specific provider or all providers
    
    Requires: Analyst role or higher
    """
    require_permission(current_user.role, "optimization", "read")
    
    try:
        task_id = trigger_optimization_analysis(request.provider_id)
        task_name = "analyze_optimization_opportunities"
        
        if request.provider_id:
            message = f"Optimization analysis triggered for provider {request.provider_id}"
        else:
            message = "Optimization analysis triggered for all providers"
        
        logger.info("Optimization analysis triggered",
                   task_id=task_id,
                   provider_id=request.provider_id,
                   user_id=current_user.id)
        
        return TaskResponse(
            task_id=task_id,
            task_name=task_name,
            status="PENDING",
            message=message,
            estimated_duration=600  # 10 minutes
        )
        
    except Exception as e:
        logger.error("Failed to trigger optimization analysis",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to trigger task: {str(e)}")

@router.post("/analyze/anomalies", response_model=TaskResponse)
async def trigger_anomaly_detection_task(
    request: TaskTriggerRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger cost anomaly detection for a specific provider or all providers
    
    Requires: Analyst role or higher
    """
    require_permission(current_user.role, "costs", "read")
    
    try:
        provider_str = str(request.provider_id) if request.provider_id else None
        task = detect_cost_anomalies.delay(provider_str)
        task_id = task.id
        task_name = "detect_cost_anomalies"
        
        if request.provider_id:
            message = f"Anomaly detection triggered for provider {request.provider_id}"
        else:
            message = "Anomaly detection triggered for all providers"
        
        logger.info("Anomaly detection triggered",
                   task_id=task_id,
                   provider_id=request.provider_id,
                   user_id=current_user.id)
        
        return TaskResponse(
            task_id=task_id,
            task_name=task_name,
            status="PENDING",
            message=message,
            estimated_duration=300  # 5 minutes
        )
        
    except Exception as e:
        logger.error("Failed to trigger anomaly detection",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to trigger task: {str(e)}")

# Maintenance Endpoints
@router.post("/maintenance/cleanup", response_model=TaskResponse)
async def trigger_data_cleanup_task(
    request: TaskTriggerRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger data cleanup task
    
    Requires: Admin role
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required")
    
    try:
        days_to_keep = request.parameters.get('days_to_keep', 365)
        task = cleanup_old_data.delay(days_to_keep)
        task_id = task.id
        task_name = "cleanup_old_data"
        message = f"Data cleanup triggered (keeping {days_to_keep} days)"
        
        logger.info("Data cleanup triggered",
                   task_id=task_id,
                   days_to_keep=days_to_keep,
                   user_id=current_user.id)
        
        return TaskResponse(
            task_id=task_id,
            task_name=task_name,
            status="PENDING",
            message=message,
            estimated_duration=1800  # 30 minutes
        )
        
    except Exception as e:
        logger.error("Failed to trigger data cleanup",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to trigger task: {str(e)}")

# Task Status and Monitoring Endpoints
@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str = Path(..., description="Task ID to check"),
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of a specific task
    
    Requires: Any authenticated user
    """
    try:
        status_info = TaskMonitor.get_task_status(task_id)
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status_info['status'],
            result=status_info['result'],
            traceback=status_info['traceback'],
            date_done=status_info['date_done']
        )
        
    except Exception as e:
        logger.error("Failed to get task status",
                    task_id=task_id,
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/active", response_model=List[ActiveTaskResponse])
async def get_active_tasks(
    current_user: User = Depends(get_current_user)
):
    """
    Get list of currently active tasks
    
    Requires: Finance Manager or Admin role
    """
    require_permission(current_user.role, "tasks", "read")
    
    try:
        active_tasks = TaskMonitor.get_active_tasks()
        
        return [
            ActiveTaskResponse(
                worker=task['worker'],
                task_id=task['task_id'],
                name=task['name'],
                args=task['args'],
                kwargs=task['kwargs'],
                time_start=task['time_start']
            )
            for task in active_tasks
        ]
        
    except Exception as e:
        logger.error("Failed to get active tasks",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get active tasks: {str(e)}")

@router.delete("/cancel/{task_id}")
async def cancel_task(
    task_id: str = Path(..., description="Task ID to cancel"),
    current_user: User = Depends(get_current_user)
):
    """
    Cancel a running task
    
    Requires: Finance Manager or Admin role
    """
    require_permission(current_user.role, "tasks", "delete")
    
    try:
        success = TaskMonitor.cancel_task(task_id)
        
        if success:
            logger.info("Task cancelled",
                       task_id=task_id,
                       user_id=current_user.id)
            return {"message": f"Task {task_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel task")
        
    except Exception as e:
        logger.error("Failed to cancel task",
                    task_id=task_id,
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@router.get("/workers/stats", response_model=WorkerStatsResponse)
async def get_worker_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get worker statistics and health information
    
    Requires: Admin role
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required")
    
    try:
        stats = TaskMonitor.get_worker_stats()
        
        return WorkerStatsResponse(
            workers=stats,
            total_workers=len(stats)
        )
        
    except Exception as e:
        logger.error("Failed to get worker stats",
                    error=str(e),
                    user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get worker stats: {str(e)}")

# Health check endpoint for task system
@router.get("/health")
async def task_system_health():
    """
    Check health of the task system
    
    Public endpoint for monitoring
    """
    try:
        # Check if we can connect to Celery
        stats = TaskMonitor.get_worker_stats()
        active_tasks = TaskMonitor.get_active_tasks()
        
        return {
            "status": "healthy",
            "workers": len(stats),
            "active_tasks": len(active_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Task system health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }