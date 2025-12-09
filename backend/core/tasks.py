"""
Background Task System for FinOps Platform
Handles automated data synchronization, monitoring, and processing
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID
import structlog
from celery import Celery
from celery.schedules import crontab
from sqlalchemy.orm import Session

from .database import get_db_session
from .models import (
    CloudProvider, CostData, Budget, BudgetAlert, 
    OptimizationRecommendation, AuditLog, RecommendationType, 
    RecommendationStatus, RiskLevel, AlertType
)
from .repositories import (
    CloudProviderRepository, CostDataRepository, 
    BudgetRepository
    # OptimizationRepository  # TODO: Implement this repository
)
from .cloud_providers import CloudProviderService
from .encryption import EncryptionService
from .alert_manager import AlertManager
from .anomaly_detector import AnomalyDetector

logger = structlog.get_logger(__name__)

# Initialize Celery app
celery_app = Celery('finops_tasks')

# Celery configuration
celery_app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression='gzip',
    result_compression='gzip',
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    # Sync cost data every 6 hours
    'sync-cost-data': {
        'task': 'backend.core.tasks.sync_all_providers_cost_data',
        'schedule': crontab(minute=0, hour='*/6'),
    },
    # Monitor budgets every hour
    'monitor-budgets': {
        'task': 'backend.core.tasks.monitor_all_budgets',
        'schedule': crontab(minute=0),
    },
    # Discover resources daily at 2 AM
    'discover-resources': {
        'task': 'backend.core.tasks.discover_all_resources',
        'schedule': crontab(hour=2, minute=0),
    },
    # Generate optimization recommendations daily at 3 AM
    'optimization-analysis': {
        'task': 'backend.core.tasks.analyze_optimization_opportunities',
        'schedule': crontab(hour=3, minute=0),
    },
    # Detect anomalies every 4 hours
    'anomaly-detection': {
        'task': 'backend.core.tasks.detect_cost_anomalies',
        'schedule': crontab(minute=0, hour='*/4'),
    },
    # Cleanup old data weekly on Sunday at 1 AM
    'cleanup-old-data': {
        'task': 'backend.core.tasks.cleanup_old_data',
        'schedule': crontab(hour=1, minute=0, day_of_week=0),
    },
}

class TaskResult:
    """Standard task result format"""
    
    def __init__(self, success: bool, message: str, data: Dict[str, Any] = None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

def get_task_dependencies():
    """Get task dependencies (repositories, services)"""
    session = next(get_db_session())
    
    # Initialize repositories
    cloud_provider_repo = CloudProviderRepository(session)
    cost_data_repo = CostDataRepository(session)
    budget_repo = BudgetRepository(session)
    # optimization_repo = OptimizationRepository(session)  # TODO: Implement this repository
    
    # Initialize services
    encryption_service = EncryptionService()
    cloud_provider_service = CloudProviderService(encryption_service)
    alert_manager = AlertManager()
    anomaly_detector = AnomalyDetector()
    
    return {
        'session': session,
        'cloud_provider_repo': cloud_provider_repo,
        'cost_data_repo': cost_data_repo,
        'budget_repo': budget_repo,
        'optimization_repo': optimization_repo,
        'cloud_provider_service': cloud_provider_service,
        'alert_manager': alert_manager,
        'anomaly_detector': anomaly_detector
    }

@celery_app.task(bind=True, max_retries=3)
def sync_provider_cost_data(self, provider_id: str, start_date: str = None, end_date: str = None):
    """
    Sync cost data for a specific cloud provider
    
    Args:
        provider_id: UUID of the cloud provider
        start_date: Start date for sync (YYYY-MM-DD), defaults to yesterday
        end_date: End date for sync (YYYY-MM-DD), defaults to today
    """
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        # Parse dates
        if not start_date:
            start_date = (date.today() - timedelta(days=1)).isoformat()
        if not end_date:
            end_date = date.today().isoformat()
        
        start_dt = datetime.fromisoformat(start_date).date()
        end_dt = datetime.fromisoformat(end_date).date()
        
        logger.info("Starting cost data sync", 
                   provider_id=provider_id,
                   start_date=start_date,
                   end_date=end_date,
                   task_id=self.request.id)
        
        # Get provider
        provider = deps['cloud_provider_repo'].get_by_id(UUID(provider_id))
        if not provider or not provider.is_active:
            raise ValueError(f"Provider {provider_id} not found or inactive")
        
        # Sync cost data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cost_data = loop.run_until_complete(
                deps['cloud_provider_service'].sync_cost_data(
                    UUID(provider_id), start_dt, end_dt
                )
            )
            
            # Save to database
            saved_records = []
            for cost_record in cost_data:
                # Check if record already exists
                existing = deps['cost_data_repo'].get_by_resource_and_date(
                    cost_record.resource_id, cost_record.cost_date
                )
                
                if existing:
                    # Update existing record
                    updated = deps['cost_data_repo'].update(
                        existing.id,
                        cost_amount=cost_record.cost_amount,
                        usage_quantity=cost_record.usage_quantity,
                        tags=cost_record.tags,
                        metadata=cost_record.metadata
                    )
                    saved_records.append(updated)
                else:
                    # Create new record
                    created = deps['cost_data_repo'].create(
                        provider_id=cost_record.provider_id,
                        resource_id=cost_record.resource_id,
                        resource_type=cost_record.resource_type,
                        service_name=cost_record.service_name,
                        cost_amount=cost_record.cost_amount,
                        currency=cost_record.currency,
                        cost_date=cost_record.cost_date,
                        usage_quantity=cost_record.usage_quantity,
                        usage_unit=cost_record.usage_unit,
                        tags=cost_record.tags,
                        metadata=cost_record.metadata
                    )
                    saved_records.append(created)
            
            # Update provider last sync time
            deps['cloud_provider_repo'].update(
                UUID(provider_id),
                last_sync=datetime.utcnow()
            )
            
            session.commit()
            
            logger.info("Cost data sync completed successfully",
                       provider_id=provider_id,
                       records_processed=len(cost_data),
                       records_saved=len(saved_records),
                       task_id=self.request.id)
            
            return TaskResult(
                success=True,
                message=f"Synced {len(saved_records)} cost records",
                data={
                    'provider_id': provider_id,
                    'records_processed': len(cost_data),
                    'records_saved': len(saved_records),
                    'date_range': f"{start_date} to {end_date}"
                }
            ).to_dict()
            
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error("Cost data sync failed",
                    provider_id=provider_id,
                    error=str(exc),
                    task_id=self.request.id)
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            retry_delay = 2 ** self.request.retries * 60  # 1, 2, 4 minutes
            logger.info("Retrying cost data sync",
                       provider_id=provider_id,
                       retry_count=self.request.retries + 1,
                       retry_delay=retry_delay)
            raise self.retry(countdown=retry_delay, exc=exc)
        
        return TaskResult(
            success=False,
            message=f"Cost data sync failed: {str(exc)}",
            data={'provider_id': provider_id, 'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True)
def sync_all_providers_cost_data(self):
    """Sync cost data for all active providers"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Starting cost data sync for all providers", task_id=self.request.id)
        
        # Get all active providers
        providers = deps['cloud_provider_repo'].get_all(filters={'is_active': True})
        
        if not providers:
            logger.info("No active providers found")
            return TaskResult(
                success=True,
                message="No active providers to sync",
                data={'provider_count': 0}
            ).to_dict()
        
        # Queue individual sync tasks
        sync_tasks = []
        for provider in providers:
            # Check if provider needs sync based on frequency
            if provider.last_sync:
                hours_since_sync = (datetime.utcnow() - provider.last_sync).total_seconds() / 3600
                if hours_since_sync < provider.sync_frequency_hours:
                    logger.debug("Skipping provider sync - too recent",
                               provider_id=provider.id,
                               hours_since_sync=hours_since_sync)
                    continue
            
            # Queue sync task
            task = sync_provider_cost_data.delay(str(provider.id))
            sync_tasks.append({
                'provider_id': str(provider.id),
                'provider_name': provider.name,
                'task_id': task.id
            })
        
        logger.info("Queued cost data sync tasks",
                   provider_count=len(providers),
                   queued_tasks=len(sync_tasks),
                   task_id=self.request.id)
        
        return TaskResult(
            success=True,
            message=f"Queued sync tasks for {len(sync_tasks)} providers",
            data={
                'total_providers': len(providers),
                'queued_tasks': len(sync_tasks),
                'tasks': sync_tasks
            }
        ).to_dict()
        
    except Exception as exc:
        logger.error("Failed to sync all providers", error=str(exc), task_id=self.request.id)
        return TaskResult(
            success=False,
            message=f"Failed to sync all providers: {str(exc)}",
            data={'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True, max_retries=2)
def discover_provider_resources(self, provider_id: str):
    """Discover resources for a specific cloud provider"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Starting resource discovery", 
                   provider_id=provider_id,
                   task_id=self.request.id)
        
        # Get provider
        provider = deps['cloud_provider_repo'].get_by_id(UUID(provider_id))
        if not provider or not provider.is_active:
            raise ValueError(f"Provider {provider_id} not found or inactive")
        
        # Discover resources
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            adapter = loop.run_until_complete(
                deps['cloud_provider_service'].get_adapter(UUID(provider_id))
            )
            
            resources = loop.run_until_complete(adapter.get_resources())
            
            # Store resource information in metadata
            resource_count = len(resources)
            resource_types = {}
            
            for resource in resources:
                resource_type = resource.get('resource_type', 'Unknown')
                resource_types[resource_type] = resource_types.get(resource_type, 0) + 1
            
            # Update provider metadata with resource discovery info
            current_metadata = provider.metadata or {}
            current_metadata.update({
                'last_resource_discovery': datetime.utcnow().isoformat(),
                'resource_count': resource_count,
                'resource_types': resource_types,
                'resources': resources[:100]  # Store first 100 resources
            })
            
            deps['cloud_provider_repo'].update(
                UUID(provider_id),
                metadata=current_metadata
            )
            
            session.commit()
            
            logger.info("Resource discovery completed",
                       provider_id=provider_id,
                       resource_count=resource_count,
                       resource_types=resource_types,
                       task_id=self.request.id)
            
            return TaskResult(
                success=True,
                message=f"Discovered {resource_count} resources",
                data={
                    'provider_id': provider_id,
                    'resource_count': resource_count,
                    'resource_types': resource_types
                }
            ).to_dict()
            
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error("Resource discovery failed",
                    provider_id=provider_id,
                    error=str(exc),
                    task_id=self.request.id)
        
        if self.request.retries < self.max_retries:
            retry_delay = 2 ** self.request.retries * 60
            raise self.retry(countdown=retry_delay, exc=exc)
        
        return TaskResult(
            success=False,
            message=f"Resource discovery failed: {str(exc)}",
            data={'provider_id': provider_id, 'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True)
def discover_all_resources(self):
    """Discover resources for all active providers"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Starting resource discovery for all providers", task_id=self.request.id)
        
        # Get all active providers
        providers = deps['cloud_provider_repo'].get_all(filters={'is_active': True})
        
        # Queue discovery tasks
        discovery_tasks = []
        for provider in providers:
            task = discover_provider_resources.delay(str(provider.id))
            discovery_tasks.append({
                'provider_id': str(provider.id),
                'provider_name': provider.name,
                'task_id': task.id
            })
        
        logger.info("Queued resource discovery tasks",
                   provider_count=len(providers),
                   task_id=self.request.id)
        
        return TaskResult(
            success=True,
            message=f"Queued discovery tasks for {len(providers)} providers",
            data={
                'provider_count': len(providers),
                'tasks': discovery_tasks
            }
        ).to_dict()
        
    except Exception as exc:
        logger.error("Failed to discover all resources", error=str(exc), task_id=self.request.id)
        return TaskResult(
            success=False,
            message=f"Failed to discover all resources: {str(exc)}",
            data={'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True)
def monitor_budget(self, budget_id: str):
    """Monitor a specific budget for threshold breaches"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Monitoring budget", budget_id=budget_id, task_id=self.request.id)
        
        # Get budget
        budget = deps['budget_repo'].get_by_id(UUID(budget_id))
        if not budget or not budget.is_active:
            raise ValueError(f"Budget {budget_id} not found or inactive")
        
        # Calculate current spend
        current_spend = deps['budget_repo'].calculate_current_spend(UUID(budget_id))
        spend_percentage = (current_spend / budget.amount) * 100 if budget.amount > 0 else 0
        
        # Check thresholds
        alerts_triggered = []
        for threshold in budget.alert_thresholds:
            if spend_percentage >= threshold:
                # Check if alert already exists for this threshold
                existing_alert = session.query(BudgetAlert).filter(
                    BudgetAlert.budget_id == budget.id,
                    BudgetAlert.threshold_percentage == threshold,
                    BudgetAlert.acknowledged == False
                ).first()
                
                if not existing_alert:
                    # Create new alert
                    alert = BudgetAlert(
                        budget_id=budget.id,
                        threshold_percentage=threshold,
                        current_spend=current_spend,
                        alert_type=AlertType.THRESHOLD,
                        triggered_at=datetime.utcnow()
                    )
                    session.add(alert)
                    alerts_triggered.append({
                        'threshold': threshold,
                        'current_spend': float(current_spend),
                        'spend_percentage': spend_percentage
                    })
        
        session.commit()
        
        # Send notifications if alerts were triggered
        if alerts_triggered and budget.notification_emails:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(
                    deps['alert_manager'].send_budget_alerts(
                        budget, alerts_triggered
                    )
                )
            finally:
                loop.close()
        
        logger.info("Budget monitoring completed",
                   budget_id=budget_id,
                   current_spend=float(current_spend),
                   spend_percentage=spend_percentage,
                   alerts_triggered=len(alerts_triggered),
                   task_id=self.request.id)
        
        return TaskResult(
            success=True,
            message=f"Budget monitored - {spend_percentage:.1f}% spent",
            data={
                'budget_id': budget_id,
                'current_spend': float(current_spend),
                'budget_amount': float(budget.amount),
                'spend_percentage': spend_percentage,
                'alerts_triggered': alerts_triggered
            }
        ).to_dict()
        
    except Exception as exc:
        logger.error("Budget monitoring failed",
                    budget_id=budget_id,
                    error=str(exc),
                    task_id=self.request.id)
        
        return TaskResult(
            success=False,
            message=f"Budget monitoring failed: {str(exc)}",
            data={'budget_id': budget_id, 'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True)
def monitor_all_budgets(self):
    """Monitor all active budgets"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Monitoring all budgets", task_id=self.request.id)
        
        # Get all active budgets
        budgets = deps['budget_repo'].get_all(filters={'is_active': True})
        
        # Queue monitoring tasks
        monitoring_tasks = []
        for budget in budgets:
            # Check if budget is in current period
            today = date.today()
            if budget.start_date <= today and (not budget.end_date or budget.end_date >= today):
                task = monitor_budget.delay(str(budget.id))
                monitoring_tasks.append({
                    'budget_id': str(budget.id),
                    'budget_name': budget.name,
                    'task_id': task.id
                })
        
        logger.info("Queued budget monitoring tasks",
                   total_budgets=len(budgets),
                   active_budgets=len(monitoring_tasks),
                   task_id=self.request.id)
        
        return TaskResult(
            success=True,
            message=f"Queued monitoring for {len(monitoring_tasks)} active budgets",
            data={
                'total_budgets': len(budgets),
                'active_budgets': len(monitoring_tasks),
                'tasks': monitoring_tasks
            }
        ).to_dict()
        
    except Exception as exc:
        logger.error("Failed to monitor all budgets", error=str(exc), task_id=self.request.id)
        return TaskResult(
            success=False,
            message=f"Failed to monitor all budgets: {str(exc)}",
            data={'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True, max_retries=2)
def analyze_optimization_opportunities(self, provider_id: str = None):
    """Analyze optimization opportunities for providers"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Analyzing optimization opportunities",
                   provider_id=provider_id,
                   task_id=self.request.id)
        
        # Get providers to analyze
        if provider_id:
            providers = [deps['cloud_provider_repo'].get_by_id(UUID(provider_id))]
        else:
            providers = deps['cloud_provider_repo'].get_all(filters={'is_active': True})
        
        total_recommendations = 0
        
        for provider in providers:
            if not provider or not provider.is_active:
                continue
            
            # Get optimization recommendations from cloud provider
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                adapter = loop.run_until_complete(
                    deps['cloud_provider_service'].get_adapter(provider.id)
                )
                
                recommendations = loop.run_until_complete(
                    adapter.get_rightsizing_recommendations()
                )
                
                # Save recommendations to database
                for rec in recommendations:
                    # Check if recommendation already exists
                    existing = deps['optimization_repo'].get_by_resource(
                        provider.id, rec['resource_id'], rec['recommendation_type']
                    )
                    
                    if existing:
                        # Update existing recommendation
                        deps['optimization_repo'].update(
                            existing.id,
                            current_cost=rec['current_cost'],
                            optimized_cost=rec['recommended_cost'],
                            potential_savings=rec['potential_savings'],
                            confidence_score=0.8,  # Default confidence
                            recommendation_text=f"Rightsize to {rec.get('recommended_instance_type', 'optimized configuration')}",
                            implementation_details=rec.get('details', {}),
                            updated_at=datetime.utcnow()
                        )
                    else:
                        # Create new recommendation
                        deps['optimization_repo'].create(
                            provider_id=provider.id,
                            resource_id=rec['resource_id'],
                            recommendation_type=RecommendationType.RIGHTSIZING,
                            current_cost=rec['current_cost'],
                            optimized_cost=rec['recommended_cost'],
                            potential_savings=rec['potential_savings'],
                            confidence_score=0.8,
                            risk_level=RiskLevel.LOW,
                            recommendation_text=f"Rightsize to {rec.get('recommended_instance_type', 'optimized configuration')}",
                            implementation_details=rec.get('details', {}),
                            status=RecommendationStatus.PENDING
                        )
                    
                    total_recommendations += 1
                
            finally:
                loop.close()
        
        session.commit()
        
        logger.info("Optimization analysis completed",
                   provider_id=provider_id,
                   total_recommendations=total_recommendations,
                   task_id=self.request.id)
        
        return TaskResult(
            success=True,
            message=f"Generated {total_recommendations} optimization recommendations",
            data={
                'provider_id': provider_id,
                'total_recommendations': total_recommendations,
                'providers_analyzed': len([p for p in providers if p and p.is_active])
            }
        ).to_dict()
        
    except Exception as exc:
        logger.error("Optimization analysis failed",
                    provider_id=provider_id,
                    error=str(exc),
                    task_id=self.request.id)
        
        if self.request.retries < self.max_retries:
            retry_delay = 2 ** self.request.retries * 60
            raise self.retry(countdown=retry_delay, exc=exc)
        
        return TaskResult(
            success=False,
            message=f"Optimization analysis failed: {str(exc)}",
            data={'provider_id': provider_id, 'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True)
def detect_cost_anomalies(self, provider_id: str = None):
    """Detect cost anomalies in spending patterns"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Detecting cost anomalies",
                   provider_id=provider_id,
                   task_id=self.request.id)
        
        # Get recent cost data for analysis
        end_date = date.today()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        filters = {'cost_date__gte': start_date, 'cost_date__lte': end_date}
        if provider_id:
            filters['provider_id'] = UUID(provider_id)
        
        cost_data = deps['cost_data_repo'].get_all(filters=filters, limit=10000)
        
        if not cost_data:
            return TaskResult(
                success=True,
                message="No cost data available for anomaly detection",
                data={'provider_id': provider_id}
            ).to_dict()
        
        # Run anomaly detection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            anomalies = loop.run_until_complete(
                deps['anomaly_detector'].detect_anomalies(cost_data)
            )
            
            # Process anomalies and create alerts if needed
            anomaly_count = len(anomalies)
            
            for anomaly in anomalies:
                # Create budget alert for anomaly
                if anomaly.get('severity', 'low') in ['high', 'critical']:
                    # Find relevant budget
                    budgets = deps['budget_repo'].get_all(filters={'is_active': True})
                    
                    for budget in budgets:
                        # Check if anomaly matches budget scope
                        if deps['budget_repo'].matches_scope(budget, anomaly):
                            alert = BudgetAlert(
                                budget_id=budget.id,
                                threshold_percentage=0,  # Anomaly alert
                                current_spend=Decimal(str(anomaly.get('amount', 0))),
                                alert_type=AlertType.ANOMALY,
                                triggered_at=datetime.utcnow()
                            )
                            session.add(alert)
                            break
            
            session.commit()
            
        finally:
            loop.close()
        
        logger.info("Anomaly detection completed",
                   provider_id=provider_id,
                   anomaly_count=anomaly_count,
                   task_id=self.request.id)
        
        return TaskResult(
            success=True,
            message=f"Detected {anomaly_count} cost anomalies",
            data={
                'provider_id': provider_id,
                'anomaly_count': anomaly_count,
                'cost_records_analyzed': len(cost_data)
            }
        ).to_dict()
        
    except Exception as exc:
        logger.error("Anomaly detection failed",
                    provider_id=provider_id,
                    error=str(exc),
                    task_id=self.request.id)
        
        return TaskResult(
            success=False,
            message=f"Anomaly detection failed: {str(exc)}",
            data={'provider_id': provider_id, 'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

@celery_app.task(bind=True)
def cleanup_old_data(self, days_to_keep: int = 365):
    """Clean up old cost data and logs"""
    try:
        deps = get_task_dependencies()
        session = deps['session']
        
        logger.info("Starting data cleanup",
                   days_to_keep=days_to_keep,
                   task_id=self.request.id)
        
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        
        # Clean up old cost data
        deleted_cost_records = deps['cost_data_repo'].delete_older_than(cutoff_date)
        
        # Clean up old audit logs (keep for compliance)
        audit_cutoff = date.today() - timedelta(days=2555)  # 7 years
        deleted_audit_records = session.query(AuditLog).filter(
            AuditLog.created_at < audit_cutoff
        ).delete()
        
        # Clean up acknowledged budget alerts older than 90 days
        alert_cutoff = date.today() - timedelta(days=90)
        deleted_alerts = session.query(BudgetAlert).filter(
            BudgetAlert.acknowledged == True,
            BudgetAlert.acknowledged_at < alert_cutoff
        ).delete()
        
        session.commit()
        
        logger.info("Data cleanup completed",
                   deleted_cost_records=deleted_cost_records,
                   deleted_audit_records=deleted_audit_records,
                   deleted_alerts=deleted_alerts,
                   task_id=self.request.id)
        
        return TaskResult(
            success=True,
            message="Data cleanup completed successfully",
            data={
                'deleted_cost_records': deleted_cost_records,
                'deleted_audit_records': deleted_audit_records,
                'deleted_alerts': deleted_alerts,
                'cutoff_date': cutoff_date.isoformat()
            }
        ).to_dict()
        
    except Exception as exc:
        logger.error("Data cleanup failed", error=str(exc), task_id=self.request.id)
        return TaskResult(
            success=False,
            message=f"Data cleanup failed: {str(exc)}",
            data={'error': str(exc)}
        ).to_dict()
    
    finally:
        if 'session' in locals():
            session.close()

# Task monitoring and management functions
class TaskMonitor:
    """Monitor and manage background tasks"""
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        result = celery_app.AsyncResult(task_id)
        return {
            'task_id': task_id,
            'status': result.status,
            'result': result.result,
            'traceback': result.traceback,
            'date_done': result.date_done
        }
    
    @staticmethod
    def get_active_tasks() -> List[Dict[str, Any]]:
        """Get list of active tasks"""
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return []
        
        tasks = []
        for worker, task_list in active_tasks.items():
            for task in task_list:
                tasks.append({
                    'worker': worker,
                    'task_id': task['id'],
                    'name': task['name'],
                    'args': task['args'],
                    'kwargs': task['kwargs'],
                    'time_start': task['time_start']
                })
        
        return tasks
    
    @staticmethod
    def cancel_task(task_id: str) -> bool:
        """Cancel a running task"""
        try:
            celery_app.control.revoke(task_id, terminate=True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_worker_stats() -> Dict[str, Any]:
        """Get worker statistics"""
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        return stats or {}

# Manual task triggers (for API endpoints)
def trigger_cost_sync(provider_id: UUID, start_date: date = None, end_date: date = None) -> str:
    """Manually trigger cost data sync"""
    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None
    
    task = sync_provider_cost_data.delay(str(provider_id), start_str, end_str)
    return task.id

def trigger_resource_discovery(provider_id: UUID) -> str:
    """Manually trigger resource discovery"""
    task = discover_provider_resources.delay(str(provider_id))
    return task.id

def trigger_budget_monitoring(budget_id: UUID) -> str:
    """Manually trigger budget monitoring"""
    task = monitor_budget.delay(str(budget_id))
    return task.id

def trigger_optimization_analysis(provider_id: UUID = None) -> str:
    """Manually trigger optimization analysis"""
    provider_str = str(provider_id) if provider_id else None
    task = analyze_optimization_opportunities.delay(provider_str)
    return task.id