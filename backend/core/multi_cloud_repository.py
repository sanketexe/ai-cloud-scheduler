"""
Multi-Cloud Cost Comparison Repository

Repository pattern implementation for multi-cloud cost comparison data access layer.
Provides specialized methods for workload specifications, cost comparisons, TCO analyses,
migration analyses, and provider pricing data with performance optimizations.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime, date, timedelta
from decimal import Decimal

from sqlalchemy import and_, or_, desc, asc, func, select, update, delete, text, Index
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound

from .repositories import BaseRepository
from .models import (
    WorkloadSpecification, MultiCloudCostComparison, TCOAnalysis, 
    MigrationAnalysis, ProviderPricing, ServiceEquivalency, 
    FeatureParityAnalysis, User
)

# Import performance metrics if available
try:
    from ..monitoring.performance_metrics import PerformanceMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

import structlog
logger = structlog.get_logger(__name__)


class MultiCloudRepository:
    """
    Comprehensive repository for multi-cloud cost comparison operations.
    
    Provides high-level methods for managing workload specifications,
    cost comparisons, TCO analyses, migration analyses, and pricing data
    with performance optimizations and metrics tracking.
    """
    
    def __init__(self, session: AsyncSession, performance_metrics: Optional['PerformanceMetrics'] = None):
        self.session = session
        self.performance_metrics = performance_metrics
        
        # Initialize individual repositories
        self.workload_repo = WorkloadSpecificationRepository(session, performance_metrics)
        self.cost_comparison_repo = CostComparisonRepository(session, performance_metrics)
        self.tco_repo = TCOAnalysisRepository(session, performance_metrics)
        self.migration_repo = MigrationAnalysisRepository(session, performance_metrics)
        self.pricing_repo = ProviderPricingRepository(session, performance_metrics)
        self.equivalency_repo = ServiceEquivalencyRepository(session, performance_metrics)
        self.parity_repo = FeatureParityRepository(session, performance_metrics)
        
        # Performance optimization settings
        self.batch_size = 1000
        self.query_timeout = 30  # seconds
        self.enable_query_cache = True
    
    def _track_query_performance(self, query_type: str, start_time: float):
        """Track database query performance metrics"""
        if self.performance_metrics and METRICS_AVAILABLE:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_metrics.track_database_query_time(query_type, duration_ms)
    
    # Workload Specification Methods
    
    async def save_workload_specification(self, workload_data: Dict[str, Any]) -> WorkloadSpecification:
        """Save a new workload specification"""
        start_time = time.time()
        try:
            result = await self.workload_repo.create(**workload_data)
            self._track_query_performance("workload_create", start_time)
            return result
        except Exception as e:
            logger.error("Failed to save workload specification", error=str(e))
            raise
    
    async def get_workload_specification(self, workload_id: UUID) -> Optional[WorkloadSpecification]:
        """Get workload specification by ID with optimized loading"""
        start_time = time.time()
        try:
            # Use joinedload for related data to avoid N+1 queries
            result = await self.session.execute(
                select(WorkloadSpecification)
                .options(joinedload(WorkloadSpecification.cost_comparisons))
                .where(WorkloadSpecification.id == workload_id)
            )
            workload = result.unique().scalar_one_or_none()
            self._track_query_performance("workload_get_by_id", start_time)
            return workload
        except Exception as e:
            logger.error("Failed to get workload specification", workload_id=workload_id, error=str(e))
            raise
    
    async def get_workload_specifications(self, user_id: UUID, limit: int = 50) -> List[WorkloadSpecification]:
        """Get workload specifications for a user with pagination"""
        start_time = time.time()
        try:
            result = await self.workload_repo.get_by_user(user_id, limit)
            self._track_query_performance("workload_get_by_user", start_time)
            return result
        except Exception as e:
            logger.error("Failed to get workload specifications", user_id=user_id, error=str(e))
            raise
    
    async def update_workload_specification(self, workload_id: UUID, **kwargs) -> Optional[WorkloadSpecification]:
        """Update workload specification"""
        return await self.workload_repo.update(workload_id, **kwargs)
    
    async def delete_workload_specification(self, workload_id: UUID) -> bool:
        """Delete workload specification"""
        return await self.workload_repo.delete(workload_id)
    
    # Cost Comparison Methods
    
    async def save_cost_comparison(self, comparison_data: Dict[str, Any]) -> MultiCloudCostComparison:
        """Save cost comparison results"""
        return await self.cost_comparison_repo.create(**comparison_data)
    
    async def get_cost_comparison(self, comparison_id: UUID) -> Optional[MultiCloudCostComparison]:
        """Get cost comparison by ID"""
        return await self.cost_comparison_repo.get_by_id(comparison_id)
    
    async def get_cost_comparisons_by_workload(self, workload_id: UUID, limit: int = 10) -> List[MultiCloudCostComparison]:
        """Get cost comparisons for a workload"""
        return await self.cost_comparison_repo.get_by_workload(workload_id, limit)
    
    async def get_latest_cost_comparison(self, workload_id: UUID) -> Optional[MultiCloudCostComparison]:
        """Get the most recent cost comparison for a workload"""
        return await self.cost_comparison_repo.get_latest_by_workload(workload_id)
    
    # TCO Analysis Methods
    
    async def save_tco_analysis(self, tco_data: Dict[str, Any]) -> TCOAnalysis:
        """Save TCO analysis results"""
        return await self.tco_repo.create(**tco_data)
    
    async def get_tco_analysis(self, analysis_id: UUID) -> Optional[TCOAnalysis]:
        """Get TCO analysis by ID"""
        return await self.tco_repo.get_by_id(analysis_id)
    
    async def get_tco_analyses_by_workload(self, workload_id: UUID, limit: int = 10) -> List[TCOAnalysis]:
        """Get TCO analyses for a workload"""
        return await self.tco_repo.get_by_workload(workload_id, limit)
    
    # Migration Analysis Methods
    
    async def save_migration_analysis(self, migration_data: Dict[str, Any]) -> MigrationAnalysis:
        """Save migration analysis results"""
        return await self.migration_repo.create(**migration_data)
    
    async def get_migration_analysis(self, analysis_id: UUID) -> Optional[MigrationAnalysis]:
        """Get migration analysis by ID"""
        return await self.migration_repo.get_by_id(analysis_id)
    
    async def get_migration_analyses_by_workload(self, workload_id: UUID, limit: int = 10) -> List[MigrationAnalysis]:
        """Get migration analyses for a workload"""
        return await self.migration_repo.get_by_workload(workload_id, limit)
    
    async def get_migration_analyses_by_providers(self, 
                                                 source_provider: str, 
                                                 target_provider: str,
                                                 limit: int = 50) -> List[MigrationAnalysis]:
        """Get migration analyses between specific providers"""
        return await self.migration_repo.get_by_providers(source_provider, target_provider, limit)
    
    # Pricing Data Methods
    
    async def save_pricing_data(self, pricing_data: List[Dict[str, Any]]) -> List[ProviderPricing]:
        """Bulk save pricing data"""
        return await self.pricing_repo.bulk_create(pricing_data)
    
    async def get_pricing_history(self, provider: str, service: str, days: int = 30) -> List[ProviderPricing]:
        """Get pricing history for a service"""
        return await self.pricing_repo.get_pricing_history(provider, service, days)
    
    async def get_current_pricing(self, provider: str, region: str, service_category: Optional[str] = None) -> List[ProviderPricing]:
        """Get current pricing for a provider and region"""
        return await self.pricing_repo.get_current_pricing(provider, region, service_category)
    
    async def update_pricing_data(self, provider: str, service: str, region: str, **kwargs) -> Optional[ProviderPricing]:
        """Update pricing data for a specific service"""
        return await self.pricing_repo.update_pricing(provider, service, region, **kwargs)
    
    # Service Equivalency Methods
    
    async def get_equivalent_services(self, source_provider: str, source_service: str) -> List[ServiceEquivalency]:
        """Get equivalent services across providers"""
        return await self.equivalency_repo.get_equivalents(source_provider, source_service)
    
    async def save_service_equivalency(self, equivalency_data: Dict[str, Any]) -> ServiceEquivalency:
        """Save service equivalency mapping"""
        return await self.equivalency_repo.create(**equivalency_data)
    
    # Analytics and Reporting Methods
    
    async def get_cost_trends(self, workload_id: UUID, days: int = 90) -> List[Dict[str, Any]]:
        """Get cost trends for a workload over time"""
        return await self.cost_comparison_repo.get_cost_trends(workload_id, days)
    
    async def get_provider_cost_summary(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get cost summary by provider for a date range"""
        return await self.cost_comparison_repo.get_provider_summary(start_date, end_date)
    
    async def get_migration_roi_analysis(self, workload_id: UUID) -> Dict[str, Any]:
        """Get ROI analysis for potential migrations"""
        return await self.migration_repo.get_roi_analysis(workload_id)
    
    # Data Retention and Cleanup Methods
    
    async def cleanup_old_comparisons(self, retention_days: int = 365) -> int:
        """Clean up old cost comparisons beyond retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        return await self.cost_comparison_repo.cleanup_old_data(cutoff_date)
    
    async def cleanup_old_pricing_data(self, retention_days: int = 90) -> int:
        """Clean up old pricing data beyond retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        return await self.pricing_repo.cleanup_old_data(cutoff_date)


class WorkloadSpecificationRepository(BaseRepository[WorkloadSpecification]):
    """Repository for WorkloadSpecification entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, WorkloadSpecification)
    
    async def get_by_user(self, user_id: UUID, limit: int = 50) -> List[WorkloadSpecification]:
        """Get workload specifications created by a user"""
        return await self.get_all(
            filters={'created_by': user_id},
            limit=limit,
            order_by='-created_at'
        )
    
    async def search_by_name(self, name_pattern: str, user_id: Optional[UUID] = None) -> List[WorkloadSpecification]:
        """Search workload specifications by name pattern"""
        try:
            query = select(WorkloadSpecification).where(
                WorkloadSpecification.name.ilike(f'%{name_pattern}%')
            )
            
            if user_id:
                query = query.where(WorkloadSpecification.created_by == user_id)
            
            query = query.order_by(WorkloadSpecification.name)
            
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise ValueError(f"Failed to search workload specifications: {str(e)}")


class CostComparisonRepository(BaseRepository[MultiCloudCostComparison]):
    """Repository for MultiCloudCostComparison entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MultiCloudCostComparison)
    
    async def get_by_workload(self, workload_id: UUID, limit: int = 10) -> List[MultiCloudCostComparison]:
        """Get cost comparisons for a workload"""
        return await self.get_all(
            filters={'workload_id': workload_id},
            limit=limit,
            order_by='-comparison_date'
        )
    
    async def get_latest_by_workload(self, workload_id: UUID) -> Optional[MultiCloudCostComparison]:
        """Get the most recent cost comparison for a workload"""
        try:
            result = await self.session.execute(
                select(MultiCloudCostComparison)
                .where(MultiCloudCostComparison.workload_id == workload_id)
                .order_by(desc(MultiCloudCostComparison.comparison_date))
                .limit(1)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise ValueError(f"Failed to get latest cost comparison: {str(e)}")
    
    async def get_cost_trends(self, workload_id: UUID, days: int = 90) -> List[Dict[str, Any]]:
        """Get cost trends for a workload over time"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            result = await self.session.execute(
                select(MultiCloudCostComparison)
                .where(
                    and_(
                        MultiCloudCostComparison.workload_id == workload_id,
                        MultiCloudCostComparison.comparison_date >= start_date
                    )
                )
                .order_by(MultiCloudCostComparison.comparison_date)
            )
            
            comparisons = result.scalars().all()
            
            return [
                {
                    'date': comp.comparison_date,
                    'aws_cost': comp.aws_monthly_cost,
                    'gcp_cost': comp.gcp_monthly_cost,
                    'azure_cost': comp.azure_monthly_cost
                }
                for comp in comparisons
            ]
        except Exception as e:
            raise ValueError(f"Failed to get cost trends: {str(e)}")
    
    async def get_provider_summary(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get cost summary by provider for a date range"""
        try:
            # This is a simplified aggregation - in production you'd use more sophisticated queries
            result = await self.session.execute(
                select(MultiCloudCostComparison)
                .where(
                    and_(
                        MultiCloudCostComparison.comparison_date >= start_date,
                        MultiCloudCostComparison.comparison_date <= end_date
                    )
                )
            )
            
            comparisons = result.scalars().all()
            
            # Aggregate costs by provider
            aws_total = sum(comp.aws_monthly_cost or 0 for comp in comparisons)
            gcp_total = sum(comp.gcp_monthly_cost or 0 for comp in comparisons)
            azure_total = sum(comp.azure_monthly_cost or 0 for comp in comparisons)
            
            return [
                {'provider': 'aws', 'total_cost': aws_total, 'comparison_count': len([c for c in comparisons if c.aws_monthly_cost])},
                {'provider': 'gcp', 'total_cost': gcp_total, 'comparison_count': len([c for c in comparisons if c.gcp_monthly_cost])},
                {'provider': 'azure', 'total_cost': azure_total, 'comparison_count': len([c for c in comparisons if c.azure_monthly_cost])}
            ]
        except Exception as e:
            raise ValueError(f"Failed to get provider summary: {str(e)}")
    
    async def cleanup_old_data(self, cutoff_date: datetime) -> int:
        """Clean up old cost comparison data"""
        try:
            result = await self.session.execute(
                delete(MultiCloudCostComparison)
                .where(MultiCloudCostComparison.comparison_date < cutoff_date)
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            raise ValueError(f"Failed to cleanup old cost comparison data: {str(e)}")


class TCOAnalysisRepository(BaseRepository[TCOAnalysis]):
    """Repository for TCOAnalysis entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TCOAnalysis)
    
    async def get_by_workload(self, workload_id: UUID, limit: int = 10) -> List[TCOAnalysis]:
        """Get TCO analyses for a workload"""
        return await self.get_all(
            filters={'workload_id': workload_id},
            limit=limit,
            order_by='-analysis_date'
        )
    
    async def get_by_time_horizon(self, time_horizon_years: int) -> List[TCOAnalysis]:
        """Get TCO analyses by time horizon"""
        return await self.get_all(
            filters={'time_horizon_years': time_horizon_years},
            order_by='-analysis_date'
        )


class MigrationAnalysisRepository(BaseRepository[MigrationAnalysis]):
    """Repository for MigrationAnalysis entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MigrationAnalysis)
    
    async def get_by_workload(self, workload_id: UUID, limit: int = 10) -> List[MigrationAnalysis]:
        """Get migration analyses for a workload"""
        return await self.get_all(
            filters={'workload_id': workload_id},
            limit=limit,
            order_by='-analysis_date'
        )
    
    async def get_by_providers(self, source_provider: str, target_provider: str, limit: int = 50) -> List[MigrationAnalysis]:
        """Get migration analyses between specific providers"""
        return await self.get_all(
            filters={'source_provider': source_provider, 'target_provider': target_provider},
            limit=limit,
            order_by='-analysis_date'
        )
    
    async def get_roi_analysis(self, workload_id: UUID) -> Dict[str, Any]:
        """Get ROI analysis for potential migrations"""
        try:
            result = await self.session.execute(
                select(MigrationAnalysis)
                .where(MigrationAnalysis.workload_id == workload_id)
                .order_by(desc(MigrationAnalysis.analysis_date))
            )
            
            analyses = result.scalars().all()
            
            if not analyses:
                return {'workload_id': workload_id, 'analyses': [], 'recommendations': []}
            
            # Calculate ROI metrics
            roi_data = []
            for analysis in analyses:
                if analysis.break_even_months:
                    roi_percentage = (12 / analysis.break_even_months) * 100 if analysis.break_even_months > 0 else 0
                    roi_data.append({
                        'source_provider': analysis.source_provider,
                        'target_provider': analysis.target_provider,
                        'migration_cost': analysis.migration_cost,
                        'break_even_months': analysis.break_even_months,
                        'roi_percentage': roi_percentage,
                        'timeline_days': analysis.migration_timeline_days
                    })
            
            # Sort by ROI percentage
            roi_data.sort(key=lambda x: x['roi_percentage'], reverse=True)
            
            return {
                'workload_id': workload_id,
                'analyses': roi_data,
                'best_roi': roi_data[0] if roi_data else None,
                'total_analyses': len(analyses)
            }
        except Exception as e:
            raise ValueError(f"Failed to get ROI analysis: {str(e)}")


class ProviderPricingRepository(BaseRepository[ProviderPricing]):
    """Repository for ProviderPricing entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ProviderPricing)
    
    async def get_pricing_history(self, provider: str, service: str, days: int = 30) -> List[ProviderPricing]:
        """Get pricing history for a service"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            result = await self.session.execute(
                select(ProviderPricing)
                .where(
                    and_(
                        ProviderPricing.provider == provider,
                        ProviderPricing.service_name == service,
                        ProviderPricing.effective_date >= start_date
                    )
                )
                .order_by(desc(ProviderPricing.effective_date))
            )
            
            return result.scalars().all()
        except Exception as e:
            raise ValueError(f"Failed to get pricing history: {str(e)}")
    
    async def get_current_pricing(self, provider: str, region: str, service_category: Optional[str] = None) -> List[ProviderPricing]:
        """Get current pricing for a provider and region"""
        try:
            # Get the most recent pricing data
            subquery = select(
                ProviderPricing.service_name,
                func.max(ProviderPricing.effective_date).label('max_date')
            ).where(
                and_(
                    ProviderPricing.provider == provider,
                    ProviderPricing.region == region
                )
            )
            
            if service_category:
                subquery = subquery.where(ProviderPricing.service_category == service_category)
            
            subquery = subquery.group_by(ProviderPricing.service_name).subquery()
            
            result = await self.session.execute(
                select(ProviderPricing)
                .join(
                    subquery,
                    and_(
                        ProviderPricing.service_name == subquery.c.service_name,
                        ProviderPricing.effective_date == subquery.c.max_date
                    )
                )
                .where(
                    and_(
                        ProviderPricing.provider == provider,
                        ProviderPricing.region == region
                    )
                )
            )
            
            return result.scalars().all()
        except Exception as e:
            raise ValueError(f"Failed to get current pricing: {str(e)}")
    
    async def update_pricing(self, provider: str, service: str, region: str, **kwargs) -> Optional[ProviderPricing]:
        """Update pricing data for a specific service"""
        try:
            # Find the most recent pricing record
            result = await self.session.execute(
                select(ProviderPricing)
                .where(
                    and_(
                        ProviderPricing.provider == provider,
                        ProviderPricing.service_name == service,
                        ProviderPricing.region == region
                    )
                )
                .order_by(desc(ProviderPricing.effective_date))
                .limit(1)
            )
            
            pricing = result.scalar_one_or_none()
            
            if pricing:
                return await self.update(pricing.id, **kwargs)
            else:
                # Create new pricing record
                return await self.create(
                    provider=provider,
                    service_name=service,
                    region=region,
                    **kwargs
                )
        except Exception as e:
            raise ValueError(f"Failed to update pricing: {str(e)}")
    
    async def cleanup_old_data(self, cutoff_date: datetime) -> int:
        """Clean up old pricing data"""
        try:
            result = await self.session.execute(
                delete(ProviderPricing)
                .where(ProviderPricing.last_updated < cutoff_date)
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            raise ValueError(f"Failed to cleanup old pricing data: {str(e)}")


class ServiceEquivalencyRepository(BaseRepository[ServiceEquivalency]):
    """Repository for ServiceEquivalency entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ServiceEquivalency)
    
    async def get_equivalents(self, source_provider: str, source_service: str) -> List[ServiceEquivalency]:
        """Get equivalent services for a source service"""
        return await self.get_all(
            filters={'source_provider': source_provider, 'source_service': source_service},
            order_by='-confidence_score'
        )
    
    async def get_by_category(self, service_category: str) -> List[ServiceEquivalency]:
        """Get service equivalencies by category"""
        return await self.get_all(
            filters={'service_category': service_category},
            order_by='-confidence_score'
        )


class FeatureParityRepository(BaseRepository[FeatureParityAnalysis]):
    """Repository for FeatureParityAnalysis entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, FeatureParityAnalysis)
    
    async def get_by_reference_service(self, reference_provider: str, reference_service: str) -> List[FeatureParityAnalysis]:
        """Get feature parity analyses for a reference service"""
        return await self.get_all(
            filters={'reference_provider': reference_provider, 'reference_service': reference_service},
            order_by='-analysis_date'
        )