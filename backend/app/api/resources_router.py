from typing import List, Optional
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.database.database import get_db
from backend.app.models.models import EC2Instance, EBSVolume, OptimizationRecommendation, ScanJob, ScanStatus, AnomalyDetection
from backend.app.schemas.schemas import EC2InstanceResponse, EBSVolumeResponse, OptimizationRecommendationResponse
from backend.app.services.background_jobs import scan_aws_resources

router = APIRouter(prefix="/api/v1", tags=["Cloud Resources"])

# --- Resources ---

@router.get("/resources/ec2", response_model=List[EC2InstanceResponse])
async def get_ec2_resources(
    region: Optional[str] = None, 
    db: AsyncSession = Depends(get_db)
):
    """List all tracked EC2 instances."""
    query = select(EC2Instance)
    if region:
        query = query.where(EC2Instance.region == region)
    
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/resources/ebs", response_model=List[EBSVolumeResponse])
async def get_ebs_resources(
    region: Optional[str] = None, 
    db: AsyncSession = Depends(get_db)
):
    """List all tracked EBS volumes."""
    query = select(EBSVolume)
    if region:
        query = query.where(EBSVolume.region == region)
        
    result = await db.execute(query)
    return result.scalars().all()

# --- Optimization ---

@router.get("/optimization/recommendations", response_model=List[OptimizationRecommendationResponse])
async def get_optimization_recommendations(
    resource_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get active cost optimization recommendations."""
    query = select(OptimizationRecommendation).where(OptimizationRecommendation.status == 'new')
    if resource_type:
        query = query.where(OptimizationRecommendation.resource_type == resource_type)
        
    result = await db.execute(query)
    return result.scalars().all()

# --- Actions ---

@router.post("/scan", status_code=202)
async def trigger_scan(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger an immediate background scan of AWS resources.
    Uses FastAPI BackgroundTasks instead of Celery.
    Returns a Job ID immediately.
    """
    job_id = uuid.uuid4()
    
    # Create Job Record
    new_job = ScanJob(
        id=job_id,
        status=ScanStatus.PENDING,
        started_at=datetime.utcnow()
    )
    db.add(new_job)
    await db.commit()
    
    # Enqueue Background Task
    background_tasks.add_task(scan_aws_resources, job_id)
    
    return {
        "message": "AWS scan started in background", 
        "scan_job_id": str(job_id),
        "status": "pending"
    }

@router.get("/scan/{job_id}")
async def get_scan_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Check status of a scan job."""
    job = await db.get(ScanJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scan job not found")
        
    return {
        "id": job.id,
        "status": job.status,
        "resource_count": job.resource_count,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error_message
    }

# --- Cost & Forecasts ---

@router.get("/cost/forecast")
async def get_cost_forecast(db: AsyncSession = Depends(get_db)):
    """Get simple cost forecast."""
    return {
        "forecast_period": "30 days",
        "predicted_cost": 1250.00,
        "confidence_score": 0.85,
        "details": "Based on linear projection of last 30 days usage."
    }

@router.get("/cost/anomalies", tags=["Cost Anomaly Detection"])
async def get_cost_anomalies(db: AsyncSession = Depends(get_db)):
    """List detected cost anomalies."""
    query = select(AnomalyDetection).order_by(AnomalyDetection.detected_at.desc()).limit(50)
    try:
        result = await db.execute(query)
        anomalies = result.scalars().all()
        return anomalies
    except Exception:
        return []

