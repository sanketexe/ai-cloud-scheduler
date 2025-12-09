"""
Data Archival and Retention System for FinOps Platform
Manages data lifecycle, archiving, and retention policies
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from dataclasses import dataclass
from enum import Enum
import structlog

from .models import CostData, AuditLog, BudgetAlert, OptimizationRecommendation
from .repositories import (
    CostDataRepository, AuditLogRepository, BudgetAlertRepository, 
    OptimizationRecommendationRepository, SystemConfigurationRepository
)
from .cache_service import CacheService
from .logging_service import LoggingService

logger = structlog.get_logger(__name__)

class RetentionPolicy(Enum):
    """Data retention policy types"""
    COST_DATA_DETAILED = "cost_data_detailed"  # 13 months
    COST_DATA_AGGREGATED = "cost_data_aggregated"  # 7 years
    AUDIT_LOGS = "audit_logs"  # 7 years
    BUDGET_ALERTS = "budget_alerts"  # 2 years
    OPTIMIZATION_RECOMMENDATIONS = "optimization_recommendations"  # 1 year
    SYSTEM_LOGS = "system_logs"  # 90 days

class ArchivalStatus(Enum):
    """Archival operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class RetentionRule:
    """Data retention rule definition"""
    policy_name: str
    retention_period_days: int
    archive_after_days: int
    delete_after_days: int
    aggregation_rules: Optional[Dict[str, Any]] = None
    enabled: bool = True

@dataclass
class ArchivalJob:
    """Data archival job"""
    job_id: str
    policy_name: str
    target_date_range: Tuple[date, date]
    records_to_archive: int
    records_archived: int
    records_deleted: int
    status: ArchivalStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class ArchivalSummary:
    """Summary of archival operations"""
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    total_records_archived: int
    total_records_deleted: int
    total_space_freed_mb: float
    policies_processed: List[str]

class DataArchivalSystem:
    """Comprehensive data archival and retention management system"""
    
    def __init__(self,
                 cost_data_repository: CostDataRepository,
                 audit_log_repository: AuditLogRepository,
                 budget_alert_repository: BudgetAlertRepository,
                 optimization_repository: OptimizationRecommendationRepository,
                 system_config_repository: SystemConfigurationRepository,
                 cache_service: CacheService,
                 logging_service: LoggingService):
        self.cost_data_repo = cost_data_repository
        self.audit_log_repo = audit_log_repository
        self.budget_alert_repo = budget_alert_repository
        self.optimization_repo = optimization_repository
        self.system_config_repo = system_config_repository
        self.cache_service = cache_service
        self.logging_service = logging_service
        
        # Default retention policies
        self.retention_policies = {
            RetentionPolicy.COST_DATA_DETAILED: RetentionRule(
                policy_name="cost_data_detailed",
                retention_period_days=395,  # 13 months
                archive_after_days=90,      # Archive after 3 months
                delete_after_days=395,      # Delete after 13 months
                aggregation_rules={
                    "aggregate_to": "monthly",
                    "preserve_fields": ["service_name", "resource_type", "tags"]
                }
            ),
            RetentionPolicy.COST_DATA_AGGREGATED: RetentionRule(
                policy_name="cost_data_aggregated",
                retention_period_days=2555,  # 7 years
                archive_after_days=365,      # Archive after 1 year
                delete_after_days=2555       # Delete after 7 years
            ),
            RetentionPolicy.AUDIT_LOGS: RetentionRule(
                policy_name="audit_logs",
                retention_period_days=2555,  # 7 years
                archive_after_days=365,      # Archive after 1 year
                delete_after_days=2555       # Delete after 7 years
            ),
            RetentionPolicy.BUDGET_ALERTS: RetentionRule(
                policy_name="budget_alerts",
                retention_period_days=730,   # 2 years
                archive_after_days=180,      # Archive after 6 months
                delete_after_days=730        # Delete after 2 years
            ),
            RetentionPolicy.OPTIMIZATION_RECOMMENDATIONS: RetentionRule(
                policy_name="optimization_recommendations",
                retention_period_days=365,   # 1 year
                archive_after_days=90,       # Archive after 3 months
                delete_after_days=365        # Delete after 1 year
            ),
            RetentionPolicy.SYSTEM_LOGS: RetentionRule(
                policy_name="system_logs",
                retention_period_days=90,    # 90 days
                archive_after_days=30,       # Archive after 30 days
                delete_after_days=90         # Delete after 90 days
            )
        }
    
    async def run_archival_process(self, policies: Optional[List[RetentionPolicy]] = None) -> ArchivalSummary:
        """
        Run the complete archival process for specified policies
        
        Args:
            policies: List of retention policies to process (all if None)
            
        Returns:
            ArchivalSummary with operation results
        """
        start_time = datetime.utcnow()
        
        if policies is None:
            policies = list(self.retention_policies.keys())
        
        self.logging_service.info(
            "Starting data archival process",
            policies=[p.value for p in policies]
        )
        
        jobs = []
        total_records_archived = 0
        total_records_deleted = 0
        successful_jobs = 0
        failed_jobs = 0
        
        try:
            for policy in policies:
                if policy not in self.retention_policies:
                    self.logging_service.warning(
                        "Unknown retention policy",
                        policy=policy.value
                    )
                    continue
                
                retention_rule = self.retention_policies[policy]
                
                if not retention_rule.enabled:
                    self.logging_service.info(
                        "Skipping disabled retention policy",
                        policy=policy.value
                    )
                    continue
                
                try:
                    job = await self._process_retention_policy(policy, retention_rule)
                    jobs.append(job)
                    
                    if job.status == ArchivalStatus.COMPLETED:
                        successful_jobs += 1
                        total_records_archived += job.records_archived
                        total_records_deleted += job.records_deleted
                    else:
                        failed_jobs += 1
                        
                except Exception as e:
                    failed_jobs += 1
                    self.logging_service.error(
                        "Error processing retention policy",
                        policy=policy.value,
                        error=str(e)
                    )
            
            # Calculate space freed (estimated)
            space_freed_mb = self._estimate_space_freed(total_records_archived, total_records_deleted)
            
            summary = ArchivalSummary(
                total_jobs=len(jobs),
                successful_jobs=successful_jobs,
                failed_jobs=failed_jobs,
                total_records_archived=total_records_archived,
                total_records_deleted=total_records_deleted,
                total_space_freed_mb=space_freed_mb,
                policies_processed=[p.value for p in policies]
            )
            
            # Cache archival summary
            await self._cache_archival_summary(summary)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.logging_service.info(
                "Data archival process completed",
                total_jobs=len(jobs),
                successful_jobs=successful_jobs,
                failed_jobs=failed_jobs,
                records_archived=total_records_archived,
                records_deleted=total_records_deleted,
                processing_time_seconds=processing_time
            )
            
            return summary
            
        except Exception as e:
            self.logging_service.error(
                "Data archival process failed",
                error=str(e)
            )
            raise
    
    async def _process_retention_policy(self, policy: RetentionPolicy, rule: RetentionRule) -> ArchivalJob:
        """Process a single retention policy"""
        job_start_time = datetime.utcnow()
        
        job = ArchivalJob(
            job_id=f"archival_{policy.value}_{job_start_time.strftime('%Y%m%d_%H%M%S')}",
            policy_name=rule.policy_name,
            target_date_range=(date.today() - timedelta(days=rule.delete_after_days), date.today()),
            records_to_archive=0,
            records_archived=0,
            records_deleted=0,
            status=ArchivalStatus.IN_PROGRESS,
            started_at=job_start_time
        )
        
        try:
            self.logging_service.info(
                "Processing retention policy",
                job_id=job.job_id,
                policy=policy.value,
                archive_after_days=rule.archive_after_days,
                delete_after_days=rule.delete_after_days
            )
            
            if policy == RetentionPolicy.COST_DATA_DETAILED:
                await self._archive_cost_data(job, rule)
            elif policy == RetentionPolicy.AUDIT_LOGS:
                await self._archive_audit_logs(job, rule)
            elif policy == RetentionPolicy.BUDGET_ALERTS:
                await self._archive_budget_alerts(job, rule)
            elif policy == RetentionPolicy.OPTIMIZATION_RECOMMENDATIONS:
                await self._archive_optimization_recommendations(job, rule)
            elif policy == RetentionPolicy.SYSTEM_LOGS:
                await self._archive_system_logs(job, rule)
            
            job.status = ArchivalStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
            self.logging_service.info(
                "Retention policy processing completed",
                job_id=job.job_id,
                policy=policy.value,
                records_archived=job.records_archived,
                records_deleted=job.records_deleted
            )
            
        except Exception as e:
            job.status = ArchivalStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            
            self.logging_service.error(
                "Retention policy processing failed",
                job_id=job.job_id,
                policy=policy.value,
                error=str(e)
            )
        
        return job
    
    async def _archive_cost_data(self, job: ArchivalJob, rule: RetentionRule) -> None:
        """Archive cost data according to retention rules"""
        try:
            current_date = date.today()
            archive_cutoff = current_date - timedelta(days=rule.archive_after_days)
            delete_cutoff = current_date - timedelta(days=rule.delete_after_days)
            
            # Get records to archive (older than archive_cutoff but newer than delete_cutoff)
            records_to_archive = await self.cost_data_repo.get_all(
                filters={
                    'cost_date__lt': archive_cutoff,
                    'cost_date__gte': delete_cutoff
                },
                limit=10000
            )
            
            job.records_to_archive = len(records_to_archive)
            
            # Archive records (in practice, this would move to cold storage)
            if records_to_archive:
                archived_data = await self._create_archived_cost_data(records_to_archive, rule)
                job.records_archived = len(archived_data)
                
                self.logging_service.info(
                    "Cost data archived",
                    job_id=job.job_id,
                    records_archived=job.records_archived
                )
            
            # Get records to delete (older than delete_cutoff)
            records_to_delete = await self.cost_data_repo.get_all(
                filters={'cost_date__lt': delete_cutoff},
                limit=10000
            )
            
            # Delete old records
            deleted_count = 0
            for record in records_to_delete:
                if await self.cost_data_repo.delete(record.id):
                    deleted_count += 1
            
            job.records_deleted = deleted_count
            
            self.logging_service.info(
                "Cost data deletion completed",
                job_id=job.job_id,
                records_deleted=deleted_count
            )
            
        except Exception as e:
            self.logging_service.error(
                "Error archiving cost data",
                job_id=job.job_id,
                error=str(e)
            )
            raise
    
    async def _archive_audit_logs(self, job: ArchivalJob, rule: RetentionRule) -> None:
        """Archive audit logs according to retention rules"""
        try:
            current_date = date.today()
            archive_cutoff = current_date - timedelta(days=rule.archive_after_days)
            delete_cutoff = current_date - timedelta(days=rule.delete_after_days)
            
            # Get records to archive
            records_to_archive = await self.audit_log_repo.get_all(
                filters={
                    'created_at__lt': datetime.combine(archive_cutoff, datetime.min.time()),
                    'created_at__gte': datetime.combine(delete_cutoff, datetime.min.time())
                },
                limit=10000
            )
            
            job.records_to_archive = len(records_to_archive)
            
            # Archive records
            if records_to_archive:
                archived_data = await self._create_archived_audit_logs(records_to_archive)
                job.records_archived = len(archived_data)
            
            # Get records to delete
            records_to_delete = await self.audit_log_repo.get_all(
                filters={'created_at__lt': datetime.combine(delete_cutoff, datetime.min.time())},
                limit=10000
            )
            
            # Delete old records
            deleted_count = 0
            for record in records_to_delete:
                if await self.audit_log_repo.delete(record.id):
                    deleted_count += 1
            
            job.records_deleted = deleted_count
            
        except Exception as e:
            self.logging_service.error(
                "Error archiving audit logs",
                job_id=job.job_id,
                error=str(e)
            )
            raise
    
    async def _archive_budget_alerts(self, job: ArchivalJob, rule: RetentionRule) -> None:
        """Archive budget alerts according to retention rules"""
        try:
            current_date = date.today()
            archive_cutoff = current_date - timedelta(days=rule.archive_after_days)
            delete_cutoff = current_date - timedelta(days=rule.delete_after_days)
            
            # Get records to archive
            records_to_archive = await self.budget_alert_repo.get_all(
                filters={
                    'created_at__lt': datetime.combine(archive_cutoff, datetime.min.time()),
                    'created_at__gte': datetime.combine(delete_cutoff, datetime.min.time())
                },
                limit=10000
            )
            
            job.records_to_archive = len(records_to_archive)
            
            # Archive records
            if records_to_archive:
                archived_data = await self._create_archived_budget_alerts(records_to_archive)
                job.records_archived = len(archived_data)
            
            # Get records to delete
            records_to_delete = await self.budget_alert_repo.get_all(
                filters={'created_at__lt': datetime.combine(delete_cutoff, datetime.min.time())},
                limit=10000
            )
            
            # Delete old records
            deleted_count = 0
            for record in records_to_delete:
                if await self.budget_alert_repo.delete(record.id):
                    deleted_count += 1
            
            job.records_deleted = deleted_count
            
        except Exception as e:
            self.logging_service.error(
                "Error archiving budget alerts",
                job_id=job.job_id,
                error=str(e)
            )
            raise
    
    async def _archive_optimization_recommendations(self, job: ArchivalJob, rule: RetentionRule) -> None:
        """Archive optimization recommendations according to retention rules"""
        try:
            current_date = date.today()
            archive_cutoff = current_date - timedelta(days=rule.archive_after_days)
            delete_cutoff = current_date - timedelta(days=rule.delete_after_days)
            
            # Get records to archive
            records_to_archive = await self.optimization_repo.get_all(
                filters={
                    'created_at__lt': datetime.combine(archive_cutoff, datetime.min.time()),
                    'created_at__gte': datetime.combine(delete_cutoff, datetime.min.time())
                },
                limit=10000
            )
            
            job.records_to_archive = len(records_to_archive)
            
            # Archive records
            if records_to_archive:
                archived_data = await self._create_archived_optimization_recommendations(records_to_archive)
                job.records_archived = len(archived_data)
            
            # Get records to delete
            records_to_delete = await self.optimization_repo.get_all(
                filters={'created_at__lt': datetime.combine(delete_cutoff, datetime.min.time())},
                limit=10000
            )
            
            # Delete old records
            deleted_count = 0
            for record in records_to_delete:
                if await self.optimization_repo.delete(record.id):
                    deleted_count += 1
            
            job.records_deleted = deleted_count
            
        except Exception as e:
            self.logging_service.error(
                "Error archiving optimization recommendations",
                job_id=job.job_id,
                error=str(e)
            )
            raise
    
    async def _archive_system_logs(self, job: ArchivalJob, rule: RetentionRule) -> None:
        """Archive system logs according to retention rules"""
        try:
            # This is a placeholder for system log archival
            # In practice, this would handle application logs, error logs, etc.
            
            job.records_to_archive = 0
            job.records_archived = 0
            job.records_deleted = 0
            
            self.logging_service.info(
                "System log archival completed (placeholder)",
                job_id=job.job_id
            )
            
        except Exception as e:
            self.logging_service.error(
                "Error archiving system logs",
                job_id=job.job_id,
                error=str(e)
            )
            raise
    
    async def _create_archived_cost_data(self, records: List[CostData], rule: RetentionRule) -> List[Dict[str, Any]]:
        """Create archived version of cost data with optional aggregation"""
        archived_data = []
        
        try:
            if rule.aggregation_rules:
                # Perform aggregation
                aggregated_data = await self._aggregate_cost_data(records, rule.aggregation_rules)
                archived_data = aggregated_data
            else:
                # Simple archival without aggregation
                for record in records:
                    archived_record = {
                        'id': str(record.id),
                        'provider_id': str(record.provider_id),
                        'resource_id': record.resource_id,
                        'resource_type': record.resource_type,
                        'service_name': record.service_name,
                        'cost_amount': float(record.cost_amount),
                        'currency': record.currency,
                        'cost_date': record.cost_date.isoformat(),
                        'usage_quantity': float(record.usage_quantity) if record.usage_quantity else None,
                        'usage_unit': record.usage_unit,
                        'tags': record.tags,
                        'metadata': record.metadata,
                        'archived_at': datetime.utcnow().isoformat()
                    }
                    archived_data.append(archived_record)
            
            # In practice, this would be stored in cold storage (S3, etc.)
            await self._store_archived_data('cost_data', archived_data)
            
            return archived_data
            
        except Exception as e:
            self.logging_service.error(
                "Error creating archived cost data",
                error=str(e)
            )
            raise
    
    async def _aggregate_cost_data(self, records: List[CostData], aggregation_rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggregate cost data according to specified rules"""
        try:
            aggregate_to = aggregation_rules.get('aggregate_to', 'monthly')
            preserve_fields = aggregation_rules.get('preserve_fields', [])
            
            # Group records by aggregation key
            aggregated_groups = {}
            
            for record in records:
                # Create aggregation key
                if aggregate_to == 'monthly':
                    date_key = record.cost_date.strftime('%Y-%m')
                elif aggregate_to == 'weekly':
                    # Get week number
                    year, week, _ = record.cost_date.isocalendar()
                    date_key = f"{year}-W{week:02d}"
                else:  # daily
                    date_key = record.cost_date.isoformat()
                
                # Create grouping key with preserved fields
                group_key_parts = [date_key, str(record.provider_id)]
                
                for field in preserve_fields:
                    if hasattr(record, field):
                        value = getattr(record, field)
                        if isinstance(value, dict):
                            # For tags, create a simplified key
                            value = str(sorted(value.items()))
                        group_key_parts.append(str(value))
                
                group_key = '|'.join(group_key_parts)
                
                if group_key not in aggregated_groups:
                    aggregated_groups[group_key] = {
                        'date_key': date_key,
                        'provider_id': str(record.provider_id),
                        'records': [],
                        'total_cost': Decimal('0'),
                        'total_usage': Decimal('0'),
                        'record_count': 0
                    }
                    
                    # Add preserved fields
                    for field in preserve_fields:
                        if hasattr(record, field):
                            aggregated_groups[group_key][field] = getattr(record, field)
                
                # Add record to group
                aggregated_groups[group_key]['records'].append(record)
                aggregated_groups[group_key]['total_cost'] += record.cost_amount
                if record.usage_quantity:
                    aggregated_groups[group_key]['total_usage'] += record.usage_quantity
                aggregated_groups[group_key]['record_count'] += 1
            
            # Create aggregated records
            aggregated_data = []
            for group_key, group_data in aggregated_groups.items():
                aggregated_record = {
                    'aggregation_key': group_key,
                    'date_key': group_data['date_key'],
                    'provider_id': group_data['provider_id'],
                    'total_cost': float(group_data['total_cost']),
                    'total_usage': float(group_data['total_usage']),
                    'record_count': group_data['record_count'],
                    'aggregation_type': aggregate_to,
                    'archived_at': datetime.utcnow().isoformat()
                }
                
                # Add preserved fields
                for field in preserve_fields:
                    if field in group_data:
                        aggregated_record[field] = group_data[field]
                
                aggregated_data.append(aggregated_record)
            
            return aggregated_data
            
        except Exception as e:
            self.logging_service.error(
                "Error aggregating cost data",
                error=str(e)
            )
            raise
    
    async def _create_archived_audit_logs(self, records: List[AuditLog]) -> List[Dict[str, Any]]:
        """Create archived version of audit logs"""
        archived_data = []
        
        for record in records:
            archived_record = {
                'id': str(record.id),
                'user_id': str(record.user_id),
                'action': record.action,
                'resource_type': record.resource_type,
                'resource_id': record.resource_id,
                'old_values': record.old_values,
                'new_values': record.new_values,
                'ip_address': record.ip_address,
                'user_agent': record.user_agent,
                'created_at': record.created_at.isoformat(),
                'archived_at': datetime.utcnow().isoformat()
            }
            archived_data.append(archived_record)
        
        await self._store_archived_data('audit_logs', archived_data)
        return archived_data
    
    async def _create_archived_budget_alerts(self, records: List[BudgetAlert]) -> List[Dict[str, Any]]:
        """Create archived version of budget alerts"""
        archived_data = []
        
        for record in records:
            archived_record = {
                'id': str(record.id),
                'budget_id': str(record.budget_id),
                'threshold_percentage': record.threshold_percentage,
                'current_spend': float(record.current_spend),
                'budget_amount': float(record.budget_amount),
                'alert_type': record.alert_type.value,
                'acknowledged': record.acknowledged,
                'created_at': record.created_at.isoformat(),
                'archived_at': datetime.utcnow().isoformat()
            }
            archived_data.append(archived_record)
        
        await self._store_archived_data('budget_alerts', archived_data)
        return archived_data
    
    async def _create_archived_optimization_recommendations(self, records: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Create archived version of optimization recommendations"""
        archived_data = []
        
        for record in records:
            archived_record = {
                'id': str(record.id),
                'provider_id': str(record.provider_id),
                'resource_id': record.resource_id,
                'recommendation_type': record.recommendation_type.value,
                'current_cost': float(record.current_cost),
                'optimized_cost': float(record.optimized_cost),
                'potential_savings': float(record.potential_savings),
                'confidence_score': float(record.confidence_score),
                'status': record.status.value,
                'created_at': record.created_at.isoformat(),
                'archived_at': datetime.utcnow().isoformat()
            }
            archived_data.append(archived_record)
        
        await self._store_archived_data('optimization_recommendations', archived_data)
        return archived_data
    
    async def _store_archived_data(self, data_type: str, archived_data: List[Dict[str, Any]]) -> None:
        """Store archived data (placeholder for actual storage implementation)"""
        try:
            # In practice, this would store data in cold storage like S3, Glacier, etc.
            storage_key = f"archived_{data_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            self.logging_service.info(
                "Archived data stored",
                data_type=data_type,
                storage_key=storage_key,
                record_count=len(archived_data)
            )
            
            # Cache metadata about archived data
            await self.cache_service.set(
                f"archive_metadata:{storage_key}",
                {
                    'data_type': data_type,
                    'record_count': len(archived_data),
                    'archived_at': datetime.utcnow().isoformat(),
                    'storage_key': storage_key
                },
                ttl=86400 * 30  # 30 days
            )
            
        except Exception as e:
            self.logging_service.error(
                "Error storing archived data",
                data_type=data_type,
                error=str(e)
            )
            raise
    
    def _estimate_space_freed(self, records_archived: int, records_deleted: int) -> float:
        """Estimate space freed by archival operations (in MB)"""
        # Rough estimates based on average record sizes
        avg_record_size_kb = 2.0  # 2KB per record average
        
        # Archived records are compressed, so they take less space
        archived_space_mb = (records_archived * avg_record_size_kb * 0.3) / 1024  # 30% of original size
        deleted_space_mb = (records_deleted * avg_record_size_kb) / 1024
        
        return archived_space_mb + deleted_space_mb
    
    async def _cache_archival_summary(self, summary: ArchivalSummary) -> None:
        """Cache archival summary for reporting"""
        try:
            cache_key = f"archival_summary:{date.today().isoformat()}"
            
            summary_data = {
                'total_jobs': summary.total_jobs,
                'successful_jobs': summary.successful_jobs,
                'failed_jobs': summary.failed_jobs,
                'total_records_archived': summary.total_records_archived,
                'total_records_deleted': summary.total_records_deleted,
                'total_space_freed_mb': summary.total_space_freed_mb,
                'policies_processed': summary.policies_processed,
                'created_at': datetime.utcnow().isoformat()
            }
            
            await self.cache_service.set(cache_key, summary_data, ttl=86400 * 7)  # 7 days
            
        except Exception as e:
            self.logging_service.warning(
                "Failed to cache archival summary",
                error=str(e)
            )
    
    async def get_retention_policy_status(self) -> Dict[str, Any]:
        """Get status of all retention policies"""
        try:
            policy_status = {}
            
            for policy, rule in self.retention_policies.items():
                current_date = date.today()
                
                # Calculate dates
                archive_cutoff = current_date - timedelta(days=rule.archive_after_days)
                delete_cutoff = current_date - timedelta(days=rule.delete_after_days)
                
                # Get record counts (simplified - would be more sophisticated in practice)
                if policy == RetentionPolicy.COST_DATA_DETAILED:
                    active_records = await self.cost_data_repo.count(
                        filters={'cost_date__gte': archive_cutoff}
                    )
                    archival_candidates = await self.cost_data_repo.count(
                        filters={
                            'cost_date__lt': archive_cutoff,
                            'cost_date__gte': delete_cutoff
                        }
                    )
                    deletion_candidates = await self.cost_data_repo.count(
                        filters={'cost_date__lt': delete_cutoff}
                    )
                else:
                    # Placeholder for other data types
                    active_records = 0
                    archival_candidates = 0
                    deletion_candidates = 0
                
                policy_status[policy.value] = {
                    'enabled': rule.enabled,
                    'retention_period_days': rule.retention_period_days,
                    'archive_after_days': rule.archive_after_days,
                    'delete_after_days': rule.delete_after_days,
                    'active_records': active_records,
                    'archival_candidates': archival_candidates,
                    'deletion_candidates': deletion_candidates,
                    'archive_cutoff_date': archive_cutoff.isoformat(),
                    'delete_cutoff_date': delete_cutoff.isoformat()
                }
            
            return {
                'policies': policy_status,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logging_service.error(
                "Error getting retention policy status",
                error=str(e)
            )
            return {'error': str(e)}
    
    async def update_retention_policy(self, policy: RetentionPolicy, **kwargs) -> bool:
        """Update a retention policy configuration"""
        try:
            if policy not in self.retention_policies:
                return False
            
            rule = self.retention_policies[policy]
            
            # Update rule attributes
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            # Save updated policy to system configuration
            await self.system_config_repo.set_config(
                key=f"retention_policy_{policy.value}",
                value={
                    'retention_period_days': rule.retention_period_days,
                    'archive_after_days': rule.archive_after_days,
                    'delete_after_days': rule.delete_after_days,
                    'enabled': rule.enabled,
                    'aggregation_rules': rule.aggregation_rules
                },
                description=f"Retention policy configuration for {policy.value}"
            )
            
            self.logging_service.info(
                "Retention policy updated",
                policy=policy.value,
                updates=kwargs
            )
            
            return True
            
        except Exception as e:
            self.logging_service.error(
                "Error updating retention policy",
                policy=policy.value,
                error=str(e)
            )
            return False