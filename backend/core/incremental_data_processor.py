"""
Incremental Data Processing System for FinOps Platform
Handles delta processing, checkpointing, and parallel processing for large datasets
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .models import CostData, CloudProvider
from .repositories import CostDataRepository, CloudProviderRepository, SystemConfigurationRepository
from .cache_service import CacheService
from .logging_service import LoggingService
from .cost_data_processor import CostDataProcessor, ProcessingResult

logger = structlog.get_logger(__name__)

class ProcessingStatus(Enum):
    """Processing job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

class ProcessingMode(Enum):
    """Data processing modes"""
    FULL_REFRESH = "full_refresh"
    INCREMENTAL = "incremental"
    DELTA_ONLY = "delta_only"
    BACKFILL = "backfill"

@dataclass
class ProcessingCheckpoint:
    """Processing checkpoint for resumability"""
    checkpoint_id: str
    job_id: str
    provider_id: UUID
    processing_date: datetime
    last_processed_date: Optional[date]
    last_processed_record_id: Optional[str]
    records_processed: int
    records_total: int
    batch_size: int
    current_batch: int
    total_batches: int
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ProcessingJob:
    """Data processing job definition"""
    job_id: str
    provider_id: UUID
    processing_mode: ProcessingMode
    start_date: date
    end_date: date
    batch_size: int
    max_parallel_batches: int
    status: ProcessingStatus
    progress_percentage: float
    records_processed: int
    records_total: int
    error_count: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_checkpoint: Optional[ProcessingCheckpoint] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchProcessingResult:
    """Result of processing a single batch"""
    batch_id: str
    job_id: str
    batch_number: int
    records_processed: int
    records_created: int
    records_updated: int
    records_skipped: int
    records_failed: int
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class IncrementalProcessingResult:
    """Result of incremental processing operation"""
    job_id: str
    provider_id: UUID
    processing_mode: ProcessingMode
    total_processing_time_seconds: float
    total_records_processed: int
    total_records_created: int
    total_records_updated: int
    total_records_skipped: int
    total_records_failed: int
    batches_processed: int
    batches_failed: int
    checkpoints_created: int
    success: bool
    error_message: Optional[str] = None
    batch_results: List[BatchProcessingResult] = field(default_factory=list)

class IncrementalDataProcessor:
    """Advanced incremental data processing with checkpointing and parallel execution"""
    
    def __init__(self,
                 cost_data_repository: CostDataRepository,
                 cloud_provider_repository: CloudProviderRepository,
                 system_config_repository: SystemConfigurationRepository,
                 cost_data_processor: CostDataProcessor,
                 cache_service: CacheService,
                 logging_service: LoggingService):
        self.cost_data_repo = cost_data_repository
        self.cloud_provider_repo = cloud_provider_repository
        self.system_config_repo = system_config_repository
        self.cost_data_processor = cost_data_processor
        self.cache_service = cache_service
        self.logging_service = logging_service
        
        # Configuration
        self.default_batch_size = 1000
        self.max_parallel_batches = 4
        self.checkpoint_interval = 100  # Create checkpoint every N batches
        self.max_retry_attempts = 3
        self.retry_delay_seconds = 60
        
        # State management
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_batches)
    
    async def process_incremental_data(self,
                                     provider_id: UUID,
                                     processing_mode: ProcessingMode = ProcessingMode.INCREMENTAL,
                                     start_date: Optional[date] = None,
                                     end_date: Optional[date] = None,
                                     batch_size: Optional[int] = None,
                                     max_parallel_batches: Optional[int] = None) -> IncrementalProcessingResult:
        """
        Process data incrementally with checkpointing and parallel execution
        
        Args:
            provider_id: UUID of the cloud provider
            processing_mode: Mode of processing (incremental, full_refresh, etc.)
            start_date: Start date for processing (auto-determined if None)
            end_date: End date for processing (defaults to today)
            batch_size: Number of records per batch
            max_parallel_batches: Maximum parallel batches
            
        Returns:
            IncrementalProcessingResult with processing statistics
        """
        job_start_time = datetime.utcnow()
        
        # Set defaults
        if not end_date:
            end_date = date.today()
        if not batch_size:
            batch_size = self.default_batch_size
        if not max_parallel_batches:
            max_parallel_batches = self.max_parallel_batches
        
        # Determine start date based on processing mode
        if not start_date:
            start_date = await self._determine_start_date(provider_id, processing_mode, end_date)
        
        # Create processing job
        job = ProcessingJob(
            job_id=str(uuid4()),
            provider_id=provider_id,
            processing_mode=processing_mode,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            max_parallel_batches=max_parallel_batches,
            status=ProcessingStatus.PENDING,
            progress_percentage=0.0,
            records_processed=0,
            records_total=0,
            error_count=0,
            created_at=job_start_time
        )
        
        self.logging_service.info(
            "Starting incremental data processing",
            job_id=job.job_id,
            provider_id=str(provider_id),
            processing_mode=processing_mode.value,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            batch_size=batch_size
        )
        
        try:
            # Register job
            with self.job_lock:
                self.active_jobs[job.job_id] = job
            
            # Check for existing checkpoint
            existing_checkpoint = await self._load_latest_checkpoint(provider_id, processing_mode)
            if existing_checkpoint and self._should_resume_from_checkpoint(existing_checkpoint, job):
                job.last_checkpoint = existing_checkpoint
                self.logging_service.info(
                    "Resuming from checkpoint",
                    job_id=job.job_id,
                    checkpoint_id=existing_checkpoint.checkpoint_id,
                    records_processed=existing_checkpoint.records_processed
                )
            
            # Start processing
            job.status = ProcessingStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            result = await self._execute_processing_job(job)
            
            # Update job status
            job.status = ProcessingStatus.COMPLETED if result.success else ProcessingStatus.FAILED
            job.completed_at = datetime.utcnow()
            
            # Clean up checkpoint if successful
            if result.success and job.last_checkpoint:
                await self._cleanup_checkpoint(job.last_checkpoint.checkpoint_id)
            
            self.logging_service.info(
                "Incremental data processing completed",
                job_id=job.job_id,
                success=result.success,
                records_processed=result.total_records_processed,
                processing_time=result.total_processing_time_seconds
            )
            
            return result
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.completed_at = datetime.utcnow()
            
            self.logging_service.error(
                "Incremental data processing failed",
                job_id=job.job_id,
                error=str(e)
            )
            
            return IncrementalProcessingResult(
                job_id=job.job_id,
                provider_id=provider_id,
                processing_mode=processing_mode,
                total_processing_time_seconds=(datetime.utcnow() - job_start_time).total_seconds(),
                total_records_processed=0,
                total_records_created=0,
                total_records_updated=0,
                total_records_skipped=0,
                total_records_failed=0,
                batches_processed=0,
                batches_failed=0,
                checkpoints_created=0,
                success=False,
                error_message=str(e)
            )
        
        finally:
            # Cleanup job
            with self.job_lock:
                self.active_jobs.pop(job.job_id, None)
    
    async def _execute_processing_job(self, job: ProcessingJob) -> IncrementalProcessingResult:
        """Execute the main processing job logic"""
        job_start_time = datetime.utcnow()
        
        try:
            # Get data to process
            data_batches = await self._prepare_data_batches(job)
            job.records_total = sum(len(batch) for batch in data_batches)
            
            if not data_batches:
                self.logging_service.info(
                    "No data to process",
                    job_id=job.job_id
                )
                return IncrementalProcessingResult(
                    job_id=job.job_id,
                    provider_id=job.provider_id,
                    processing_mode=job.processing_mode,
                    total_processing_time_seconds=0,
                    total_records_processed=0,
                    total_records_created=0,
                    total_records_updated=0,
                    total_records_skipped=0,
                    total_records_failed=0,
                    batches_processed=0,
                    batches_failed=0,
                    checkpoints_created=0,
                    success=True
                )
            
            # Process batches
            batch_results = []
            checkpoints_created = 0
            
            # Determine starting batch if resuming from checkpoint
            start_batch = 0
            if job.last_checkpoint:
                start_batch = job.last_checkpoint.current_batch
                job.records_processed = job.last_checkpoint.records_processed
            
            # Process batches in parallel
            semaphore = asyncio.Semaphore(job.max_parallel_batches)
            
            async def process_batch_with_semaphore(batch_data, batch_number):
                async with semaphore:
                    return await self._process_single_batch(job, batch_data, batch_number)
            
            # Create tasks for remaining batches
            tasks = []
            for i in range(start_batch, len(data_batches)):
                task = asyncio.create_task(
                    process_batch_with_semaphore(data_batches[i], i)
                )
                tasks.append(task)
            
            # Process batches and collect results
            for completed_task in asyncio.as_completed(tasks):
                try:
                    batch_result = await completed_task
                    batch_results.append(batch_result)
                    
                    # Update job progress
                    job.records_processed += batch_result.records_processed
                    job.progress_percentage = (job.records_processed / job.records_total) * 100
                    
                    # Create checkpoint periodically
                    if len(batch_results) % self.checkpoint_interval == 0:
                        checkpoint = await self._create_checkpoint(job, batch_result.batch_number)
                        job.last_checkpoint = checkpoint
                        checkpoints_created += 1
                    
                    self.logging_service.debug(
                        "Batch processing completed",
                        job_id=job.job_id,
                        batch_number=batch_result.batch_number,
                        records_processed=batch_result.records_processed,
                        progress=job.progress_percentage
                    )
                    
                except Exception as e:
                    self.logging_service.error(
                        "Batch processing failed",
                        job_id=job.job_id,
                        error=str(e)
                    )
                    job.error_count += 1
            
            # Calculate totals
            total_processing_time = (datetime.utcnow() - job_start_time).total_seconds()
            total_records_created = sum(r.records_created for r in batch_results)
            total_records_updated = sum(r.records_updated for r in batch_results)
            total_records_skipped = sum(r.records_skipped for r in batch_results)
            total_records_failed = sum(r.records_failed for r in batch_results)
            batches_failed = len([r for r in batch_results if not r.success])
            
            success = batches_failed == 0 and job.error_count == 0
            
            # Update last processed date
            if success:
                await self._update_last_processed_date(job.provider_id, job.processing_mode, job.end_date)
            
            return IncrementalProcessingResult(
                job_id=job.job_id,
                provider_id=job.provider_id,
                processing_mode=job.processing_mode,
                total_processing_time_seconds=total_processing_time,
                total_records_processed=job.records_processed,
                total_records_created=total_records_created,
                total_records_updated=total_records_updated,
                total_records_skipped=total_records_skipped,
                total_records_failed=total_records_failed,
                batches_processed=len(batch_results),
                batches_failed=batches_failed,
                checkpoints_created=checkpoints_created,
                success=success,
                batch_results=batch_results
            )
            
        except Exception as e:
            self.logging_service.error(
                "Processing job execution failed",
                job_id=job.job_id,
                error=str(e)
            )
            raise
    
    async def _prepare_data_batches(self, job: ProcessingJob) -> List[List[Dict[str, Any]]]:
        """Prepare data batches for processing"""
        try:
            # Get raw data from cloud provider (mock implementation)
            raw_data = await self._fetch_raw_data(job.provider_id, job.start_date, job.end_date)
            
            # Filter data based on processing mode
            if job.processing_mode == ProcessingMode.INCREMENTAL:
                raw_data = await self._filter_incremental_data(raw_data, job)
            elif job.processing_mode == ProcessingMode.DELTA_ONLY:
                raw_data = await self._filter_delta_data(raw_data, job)
            
            # Split into batches
            batches = []
            for i in range(0, len(raw_data), job.batch_size):
                batch = raw_data[i:i + job.batch_size]
                batches.append(batch)
            
            self.logging_service.info(
                "Prepared data batches",
                job_id=job.job_id,
                total_records=len(raw_data),
                total_batches=len(batches),
                batch_size=job.batch_size
            )
            
            return batches
            
        except Exception as e:
            self.logging_service.error(
                "Error preparing data batches",
                job_id=job.job_id,
                error=str(e)
            )
            raise
    
    async def _process_single_batch(self, job: ProcessingJob, batch_data: List[Dict[str, Any]], batch_number: int) -> BatchProcessingResult:
        """Process a single batch of data"""
        batch_start_time = datetime.utcnow()
        batch_id = f"{job.job_id}_batch_{batch_number}"
        
        try:
            self.logging_service.debug(
                "Processing batch",
                job_id=job.job_id,
                batch_id=batch_id,
                batch_number=batch_number,
                records_count=len(batch_data)
            )
            
            # Process the batch using the cost data processor
            processing_result = await self.cost_data_processor.process_cost_data(
                job.provider_id, batch_data
            )
            
            processing_time = (datetime.utcnow() - batch_start_time).total_seconds()
            
            return BatchProcessingResult(
                batch_id=batch_id,
                job_id=job.job_id,
                batch_number=batch_number,
                records_processed=len(batch_data),
                records_created=processing_result.created_count,
                records_updated=processing_result.updated_count,
                records_skipped=processing_result.skipped_count,
                records_failed=processing_result.error_count,
                processing_time_seconds=processing_time,
                success=processing_result.success
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - batch_start_time).total_seconds()
            
            self.logging_service.error(
                "Batch processing failed",
                job_id=job.job_id,
                batch_id=batch_id,
                batch_number=batch_number,
                error=str(e)
            )
            
            return BatchProcessingResult(
                batch_id=batch_id,
                job_id=job.job_id,
                batch_number=batch_number,
                records_processed=0,
                records_created=0,
                records_updated=0,
                records_skipped=0,
                records_failed=len(batch_data),
                processing_time_seconds=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _fetch_raw_data(self, provider_id: UUID, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Fetch raw data from cloud provider (mock implementation)"""
        # This is a mock implementation - in practice, this would call the actual cloud provider APIs
        mock_data = []
        
        current_date = start_date
        while current_date <= end_date:
            # Generate mock cost records for each day
            for i in range(10):  # 10 records per day
                mock_data.append({
                    'provider_id': str(provider_id),
                    'resource_id': f'resource-{i:03d}',
                    'resource_type': 'EC2-Instance',
                    'service_name': 'Amazon EC2',
                    'cost_amount': float(Decimal('10.50') + Decimal(str(i * 0.1))),
                    'currency': 'USD',
                    'cost_date': current_date.isoformat(),
                    'usage_quantity': 24.0,
                    'usage_unit': 'Hours',
                    'tags': {'Environment': 'Production', 'Team': f'Team-{i % 3}'},
                    'metadata': {'region': 'us-east-1', 'instance_type': 't3.medium'}
                })
            
            current_date += timedelta(days=1)
        
        return mock_data
    
    async def _filter_incremental_data(self, raw_data: List[Dict[str, Any]], job: ProcessingJob) -> List[Dict[str, Any]]:
        """Filter data for incremental processing"""
        # Get last processed date
        last_processed_date = await self._get_last_processed_date(job.provider_id, job.processing_mode)
        
        if not last_processed_date:
            return raw_data  # No previous processing, include all data
        
        # Filter out data that was already processed
        filtered_data = []
        for record in raw_data:
            record_date = datetime.fromisoformat(record['cost_date']).date()
            if record_date > last_processed_date:
                filtered_data.append(record)
        
        return filtered_data
    
    async def _filter_delta_data(self, raw_data: List[Dict[str, Any]], job: ProcessingJob) -> List[Dict[str, Any]]:
        """Filter data for delta-only processing"""
        # For delta processing, we need to identify what has changed
        # This is a simplified implementation - in practice, you'd compare with existing data
        
        # Get existing data hashes
        existing_hashes = await self._get_existing_data_hashes(job.provider_id, job.start_date, job.end_date)
        
        delta_data = []
        for record in raw_data:
            # Create hash for the record
            record_hash = self._create_record_hash(record)
            
            # Include if hash doesn't exist (new) or is different (changed)
            if record_hash not in existing_hashes:
                delta_data.append(record)
        
        return delta_data
    
    def _create_record_hash(self, record: Dict[str, Any]) -> str:
        """Create a hash for a data record to detect changes"""
        # Create hash based on key fields
        hash_fields = {
            'provider_id': record.get('provider_id'),
            'resource_id': record.get('resource_id'),
            'cost_date': record.get('cost_date'),
            'cost_amount': record.get('cost_amount'),
            'usage_quantity': record.get('usage_quantity')
        }
        
        hash_string = json.dumps(hash_fields, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    async def _get_existing_data_hashes(self, provider_id: UUID, start_date: date, end_date: date) -> Set[str]:
        """Get hashes of existing data records"""
        try:
            # Get existing cost data
            existing_data = await self.cost_data_repo.get_cost_data_by_date_range(
                provider_id, start_date, end_date
            )
            
            # Create hashes
            hashes = set()
            for record in existing_data:
                record_dict = {
                    'provider_id': str(record.provider_id),
                    'resource_id': record.resource_id,
                    'cost_date': record.cost_date.isoformat(),
                    'cost_amount': float(record.cost_amount),
                    'usage_quantity': float(record.usage_quantity) if record.usage_quantity else None
                }
                hashes.add(self._create_record_hash(record_dict))
            
            return hashes
            
        except Exception as e:
            self.logging_service.error(
                "Error getting existing data hashes",
                provider_id=str(provider_id),
                error=str(e)
            )
            return set()
    
    async def _determine_start_date(self, provider_id: UUID, processing_mode: ProcessingMode, end_date: date) -> date:
        """Determine the start date for processing based on mode and history"""
        if processing_mode == ProcessingMode.FULL_REFRESH:
            # Full refresh starts from provider creation or 90 days ago
            provider = await self.cloud_provider_repo.get_by_id(provider_id)
            if provider:
                provider_start = provider.created_at.date()
                return max(provider_start, end_date - timedelta(days=90))
            else:
                return end_date - timedelta(days=90)
        
        elif processing_mode in [ProcessingMode.INCREMENTAL, ProcessingMode.DELTA_ONLY]:
            # Incremental starts from last processed date
            last_processed = await self._get_last_processed_date(provider_id, processing_mode)
            if last_processed:
                return last_processed + timedelta(days=1)
            else:
                # No previous processing, start from 7 days ago
                return end_date - timedelta(days=7)
        
        elif processing_mode == ProcessingMode.BACKFILL:
            # Backfill starts from 30 days ago
            return end_date - timedelta(days=30)
        
        else:
            # Default to 7 days ago
            return end_date - timedelta(days=7)
    
    async def _get_last_processed_date(self, provider_id: UUID, processing_mode: ProcessingMode) -> Optional[date]:
        """Get the last processed date for a provider and processing mode"""
        try:
            config_key = f"last_processed_date_{provider_id}_{processing_mode.value}"
            config = await self.system_config_repo.get_by_key(config_key)
            
            if config and config.value:
                return datetime.fromisoformat(config.value).date()
            
            return None
            
        except Exception as e:
            self.logging_service.error(
                "Error getting last processed date",
                provider_id=str(provider_id),
                processing_mode=processing_mode.value,
                error=str(e)
            )
            return None
    
    async def _update_last_processed_date(self, provider_id: UUID, processing_mode: ProcessingMode, processed_date: date) -> None:
        """Update the last processed date for a provider and processing mode"""
        try:
            config_key = f"last_processed_date_{provider_id}_{processing_mode.value}"
            await self.system_config_repo.set_config(
                key=config_key,
                value=processed_date.isoformat(),
                description=f"Last processed date for {processing_mode.value} processing"
            )
            
        except Exception as e:
            self.logging_service.error(
                "Error updating last processed date",
                provider_id=str(provider_id),
                processing_mode=processing_mode.value,
                error=str(e)
            )
    
    async def _create_checkpoint(self, job: ProcessingJob, current_batch: int) -> ProcessingCheckpoint:
        """Create a processing checkpoint"""
        checkpoint = ProcessingCheckpoint(
            checkpoint_id=str(uuid4()),
            job_id=job.job_id,
            provider_id=job.provider_id,
            processing_date=datetime.utcnow(),
            last_processed_date=job.end_date,
            last_processed_record_id=None,  # Could be enhanced to track specific records
            records_processed=job.records_processed,
            records_total=job.records_total,
            batch_size=job.batch_size,
            current_batch=current_batch,
            total_batches=job.records_total // job.batch_size + (1 if job.records_total % job.batch_size else 0),
            checkpoint_data={
                'processing_mode': job.processing_mode.value,
                'start_date': job.start_date.isoformat(),
                'end_date': job.end_date.isoformat()
            }
        )
        
        # Cache the checkpoint
        await self._save_checkpoint(checkpoint)
        
        self.logging_service.info(
            "Created processing checkpoint",
            job_id=job.job_id,
            checkpoint_id=checkpoint.checkpoint_id,
            current_batch=current_batch,
            records_processed=job.records_processed
        )
        
        return checkpoint
    
    async def _save_checkpoint(self, checkpoint: ProcessingCheckpoint) -> None:
        """Save checkpoint to cache"""
        try:
            cache_key = f"processing_checkpoint:{checkpoint.provider_id}:{checkpoint.job_id}"
            
            checkpoint_data = {
                'checkpoint_id': checkpoint.checkpoint_id,
                'job_id': checkpoint.job_id,
                'provider_id': str(checkpoint.provider_id),
                'processing_date': checkpoint.processing_date.isoformat(),
                'last_processed_date': checkpoint.last_processed_date.isoformat() if checkpoint.last_processed_date else None,
                'records_processed': checkpoint.records_processed,
                'records_total': checkpoint.records_total,
                'batch_size': checkpoint.batch_size,
                'current_batch': checkpoint.current_batch,
                'total_batches': checkpoint.total_batches,
                'checkpoint_data': checkpoint.checkpoint_data,
                'created_at': checkpoint.created_at.isoformat()
            }
            
            await self.cache_service.set(cache_key, checkpoint_data, ttl=86400 * 7)  # 7 days
            
        except Exception as e:
            self.logging_service.error(
                "Error saving checkpoint",
                checkpoint_id=checkpoint.checkpoint_id,
                error=str(e)
            )
    
    async def _load_latest_checkpoint(self, provider_id: UUID, processing_mode: ProcessingMode) -> Optional[ProcessingCheckpoint]:
        """Load the latest checkpoint for a provider and processing mode"""
        try:
            # Look for checkpoints in cache
            cache_pattern = f"processing_checkpoint:{provider_id}:*"
            
            # This is a simplified implementation - in practice, you'd search for the latest checkpoint
            # For now, we'll return None to indicate no checkpoint found
            return None
            
        except Exception as e:
            self.logging_service.error(
                "Error loading checkpoint",
                provider_id=str(provider_id),
                processing_mode=processing_mode.value,
                error=str(e)
            )
            return None
    
    def _should_resume_from_checkpoint(self, checkpoint: ProcessingCheckpoint, job: ProcessingJob) -> bool:
        """Determine if processing should resume from a checkpoint"""
        # Check if checkpoint is recent (within last 24 hours)
        if (datetime.utcnow() - checkpoint.created_at).total_seconds() > 86400:
            return False
        
        # Check if checkpoint matches current job parameters
        if (checkpoint.provider_id != job.provider_id or
            checkpoint.checkpoint_data.get('processing_mode') != job.processing_mode.value):
            return False
        
        # Check if checkpoint is not already completed
        if checkpoint.records_processed >= checkpoint.records_total:
            return False
        
        return True
    
    async def _cleanup_checkpoint(self, checkpoint_id: str) -> None:
        """Clean up a checkpoint after successful processing"""
        try:
            # Remove from cache
            cache_keys = await self.cache_service.get_keys_by_pattern(f"*{checkpoint_id}*")
            for key in cache_keys:
                await self.cache_service.delete(key)
            
            self.logging_service.info(
                "Cleaned up checkpoint",
                checkpoint_id=checkpoint_id
            )
            
        except Exception as e:
            self.logging_service.warning(
                "Error cleaning up checkpoint",
                checkpoint_id=checkpoint_id,
                error=str(e)
            )
    
    async def get_processing_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get the status of a processing job"""
        with self.job_lock:
            return self.active_jobs.get(job_id)
    
    async def cancel_processing_job(self, job_id: str) -> bool:
        """Cancel a running processing job"""
        try:
            with self.job_lock:
                job = self.active_jobs.get(job_id)
                if job and job.status == ProcessingStatus.RUNNING:
                    job.status = ProcessingStatus.CANCELLED
                    
                    self.logging_service.info(
                        "Processing job cancelled",
                        job_id=job_id
                    )
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logging_service.error(
                "Error cancelling processing job",
                job_id=job_id,
                error=str(e)
            )
            return False
    
    async def get_processing_metrics(self, provider_id: UUID, days: int = 7) -> Dict[str, Any]:
        """Get processing metrics for a provider over the last N days"""
        try:
            # This would typically query a metrics store or database
            # For now, return mock metrics
            
            return {
                "provider_id": str(provider_id),
                "period_days": days,
                "total_jobs": 5,
                "successful_jobs": 4,
                "failed_jobs": 1,
                "total_records_processed": 50000,
                "average_processing_time_seconds": 120.5,
                "average_batch_size": 1000,
                "checkpoints_created": 15,
                "last_processing_date": date.today().isoformat()
            }
            
        except Exception as e:
            self.logging_service.error(
                "Error getting processing metrics",
                provider_id=str(provider_id),
                error=str(e)
            )
            return {"error": str(e)}