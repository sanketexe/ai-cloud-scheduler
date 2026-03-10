from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from uuid import UUID
import boto3

from .auth import get_current_user
from .models import User
from .scheduler_service import SchedulerService
from pydantic import BaseModel

router = APIRouter(
    prefix="/scheduler",
    tags=["scheduler"]
)

class ScheduleCreateRequest(BaseModel):
    instance_id: str
    instance_name: Optional[str] = None
    schedule_type: str = "manual"
    start_time: str = "08:00"
    stop_time: str = "20:00"
    days_of_week: Optional[List[str]] = None
    enabled: bool = True
    estimated_monthly_savings: float = 0.0

class ScheduleUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    schedule_type: Optional[str] = None
    start_time: Optional[str] = None
    stop_time: Optional[str] = None
    days: Optional[List[str]] = None
    estimated_monthly_savings: Optional[float] = None

def get_scheduler_service():
    session = boto3.Session(region_name="us-east-1")
    return SchedulerService(boto3_session=session)

@router.get("/resources")
async def get_schedulable_resources(current_user: User = Depends(get_current_user)):
    """Get resources that can be scheduled (EC2, RDS)"""
    service = get_scheduler_service()
    return service.get_schedulable_resources()

@router.post("/analyze/{instance_id}")
async def analyze_instance_metrics(
    instance_id: str,
    current_user: User = Depends(get_current_user)
):
    """Analyze an instance's metrics to generate an optimal schedule"""
    service = get_scheduler_service()
    try:
        recommendation = service.analyze_resource(instance_id)
        if recommendation and "error" in recommendation.get("analysis", {}):
             raise HTTPException(status_code=500, detail=recommendation["analysis"]["error"])
        return recommendation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/schedules")
async def get_schedules(current_user: User = Depends(get_current_user)):
    """List all configured schedules"""
    service = get_scheduler_service()
    return service.get_schedules()

@router.post("/schedules")
async def create_schedule(
    request: ScheduleCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new resource schedule"""
    service = get_scheduler_service()
    try:
        schedule_data = request.dict(exclude_unset=True)
        # Rename 'days_of_week' to 'days' if present
        if "days_of_week" in schedule_data:
             schedule_data["days"] = schedule_data.pop("days_of_week")
             
        schedule = service.create_schedule(schedule_data)
        return schedule
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create schedule: {str(e)}")

@router.put("/schedules/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    request: ScheduleUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update an existing schedule (e.g. pause/resume)"""
    service = get_scheduler_service()
    try:
        updates = request.dict(exclude_unset=True)
        schedule = service.update_schedule(
            schedule_id=schedule_id,
            data=updates
        )
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        return schedule
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update schedule: {str(e)}")

@router.delete("/schedules/{schedule_id}")
async def delete_schedule(
    schedule_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a schedule"""
    service = get_scheduler_service()
    success = service.delete_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found or could not be deleted")
    return {"status": "deleted"}

@router.get("/savings")
async def get_projected_savings(current_user: User = Depends(get_current_user)):
    """Compute projected monthly savings from all active schedules"""
    service = get_scheduler_service()
    summary = service.get_savings_summary()
    return {"projected_monthly_savings": summary.get("estimated_monthly_savings", 0.0)}
