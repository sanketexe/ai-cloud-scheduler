import asyncio
import logging
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from sqlalchemy import select

from backend.app.database.database import AsyncSessionLocal
from backend.app.models.models import ScanJob, ScanStatus, EC2Instance, EBSVolume, OptimizationRecommendation
from backend.app.aws.aws_collector import AWSCollector
from backend.app.optimizers.resource_optimizer import ResourceOptimizer

logger = logging.getLogger(__name__)

# Thread pool for blocking I/O (AWS API calls)
batch_executor = ThreadPoolExecutor(max_workers=3)

async def scan_aws_resources(scan_id: uuid.UUID):
    """
    Background task to scan AWS resources.
    Executes blocking AWS calls in a thread pool to avoid blocking the asyncio loop.
    """
    logger.info(f"Starting scan job {scan_id}")
    
    async with AsyncSessionLocal() as session:
        # Update status to RUNNING
        job = await session.get(ScanJob, scan_id)
        if not job:
            logger.error(f"Scan job {scan_id} not found in DB")
            return
            
        job.status = ScanStatus.RUNNING
        await session.commit()

        try:
            # 1. Collect Data (Offload Sync -> ThreadPool)
            loop = asyncio.get_running_loop()
            collector = AWSCollector(region="us-east-1") 
            
            logger.info("Fetching AWS resources...")
            data = await loop.run_in_executor(batch_executor, collector.collect_resources)
            
            instances = data.get("instances", [])
            volumes = data.get("volumes", [])
            
            # 2. Store Resources (Async DB)
            # EC2 Instances
            for inst in instances:
                instance_id = inst["instance_id"]
                existing_inst = await session.get(EC2Instance, instance_id)
                
                if not existing_inst:
                    new_inst = EC2Instance(
                        id=instance_id, 
                        instance_type=inst["instance_type"],
                        state=inst["state"],
                        region=inst["region"]
                        # launch_time mapping logic if needed
                    )
                    session.add(new_inst)
                else:
                    existing_inst.state = inst["state"]
            
            # EBS Volumes
            for vol in volumes:
                volume_id = vol["volume_id"]
                existing_vol = await session.get(EBSVolume, volume_id)
                
                if not existing_vol:
                    new_vol = EBSVolume(
                        id=volume_id,
                        size=vol["size_gb"],
                        volume_type=vol["volume_type"],
                        state=vol["state"],
                        region=vol["region"]
                    )
                    session.add(new_vol)
                else:
                    existing_vol.state = vol["state"]
            
            # 3. Optimize (Offload Sync -> ThreadPool)
            logger.info("Running optimization analysis...")
            optimizer = ResourceOptimizer()
            recs = await loop.run_in_executor(batch_executor, optimizer.optimize, data)
            
            # 4. Save Recommendations (Async DB)
            for r in recs:
                new_rec = OptimizationRecommendation(
                    resource_id=r["resource_id"],
                    resource_type=r["resource_type"],
                    recommendation_type=r["issue"], # Mapping issue -> recommendation_type
                    # Mapped fields:
                    description=r["details"],
                    potential_savings=r["estimated_savings"],
                    confidence_score=0.9,
                    status="new",
                    generated_at=datetime.utcnow()
                )
                session.add(new_rec)
            
            # Update Job Status
            job.status = ScanStatus.COMPLETED
            job.resource_count = len(instances) + len(volumes)
            job.completed_at = datetime.utcnow()
            await session.commit()
            
            logger.info(f"Scan job {scan_id} completed successfully.")

        except Exception as e:
            logger.exception(f"Scan job {scan_id} failed")
            job.status = ScanStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await session.commit()
