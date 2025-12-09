"""
Data Quality Checker for FinOps Platform
Provides comprehensive data quality assessment and reporting
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from dataclasses import dataclass
from enum import Enum
import statistics
import structlog

from .models import CostData, CloudProvider
from .repositories import CostDataRepository, CloudProviderRepository
from .cost_data_processor import DataQualityIssueType, ValidationSeverity

logger = structlog.get_logger(__name__)

class DataQualityMetric(Enum):
    """Data quality metrics"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"

@dataclass
class QualityScore:
    """Data quality score for a specific metric"""
    metric: DataQualityMetric
    score: float  # 0.0 to 1.0
    total_records: int
    passed_records: int
    failed_records: int
    issues: List[Dict[str, Any]]
    
    @property
    def percentage(self) -> float:
        """Get score as percentage"""
        return self.score * 100

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    provider_id: UUID
    provider_name: str
    assessment_date: datetime
    date_range: Tuple[date, date]
    total_records: int
    overall_score: float
    metric_scores: Dict[DataQualityMetric, QualityScore]
    critical_issues: List[Dict[str, Any]]
    recommendations: List[str]
    
    def get_grade(self) -> str:
        """Get letter grade based on overall score"""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.90:
            return "A"
        elif self.overall_score >= 0.85:
            return "B+"
        elif self.overall_score >= 0.80:
            return "B"
        elif self.overall_score >= 0.75:
            return "C+"
        elif self.overall_score >= 0.70:
            return "C"
        elif self.overall_score >= 0.60:
            return "D"
        else:
            return "F"

class DataQualityChecker:
    """Comprehensive data quality assessment for cost data"""
    
    def __init__(self, 
                 cost_data_repository: CostDataRepository,
                 cloud_provider_repository: CloudProviderRepository):
        self.cost_data_repo = cost_data_repository
        self.cloud_provider_repo = cloud_provider_repository
    
    async def assess_data_quality(self, 
                                provider_id: UUID, 
                                start_date: date = None, 
                                end_date: date = None) -> DataQualityReport:
        """
        Perform comprehensive data quality assessment
        
        Args:
            provider_id: UUID of the cloud provider
            start_date: Start date for assessment (defaults to 30 days ago)
            end_date: End date for assessment (defaults to today)
            
        Returns:
            DataQualityReport with detailed quality metrics
        """
        # Set default date range
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        logger.info(
            "Starting data quality assessment",
            provider_id=str(provider_id),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        # Get provider information
        provider = await self.cloud_provider_repo.get_by_id(provider_id)
        if not provider:
            raise ValueError(f"Provider {provider_id} not found")
        
        # Get cost data for assessment
        cost_data = await self.cost_data_repo.get_cost_data_by_date_range(
            provider_id, start_date, end_date
        )
        
        if not cost_data:
            logger.warning(
                "No cost data found for quality assessment",
                provider_id=str(provider_id),
                date_range=f"{start_date} to {end_date}"
            )
            
            # Return empty report
            return DataQualityReport(
                provider_id=provider_id,
                provider_name=provider.name,
                assessment_date=datetime.utcnow(),
                date_range=(start_date, end_date),
                total_records=0,
                overall_score=0.0,
                metric_scores={},
                critical_issues=[],
                recommendations=["No data available for assessment"]
            )
        
        # Assess each quality metric
        metric_scores = {}
        
        # Completeness assessment
        metric_scores[DataQualityMetric.COMPLETENESS] = await self._assess_completeness(cost_data)
        
        # Accuracy assessment
        metric_scores[DataQualityMetric.ACCURACY] = await self._assess_accuracy(cost_data)
        
        # Consistency assessment
        metric_scores[DataQualityMetric.CONSISTENCY] = await self._assess_consistency(cost_data)
        
        # Timeliness assessment
        metric_scores[DataQualityMetric.TIMELINESS] = await self._assess_timeliness(cost_data, end_date)
        
        # Validity assessment
        metric_scores[DataQualityMetric.VALIDITY] = await self._assess_validity(cost_data)
        
        # Uniqueness assessment
        metric_scores[DataQualityMetric.UNIQUENESS] = await self._assess_uniqueness(cost_data)
        
        # Calculate overall score (weighted average)
        weights = {
            DataQualityMetric.COMPLETENESS: 0.25,
            DataQualityMetric.ACCURACY: 0.25,
            DataQualityMetric.CONSISTENCY: 0.15,
            DataQualityMetric.TIMELINESS: 0.15,
            DataQualityMetric.VALIDITY: 0.10,
            DataQualityMetric.UNIQUENESS: 0.10
        }
        
        overall_score = sum(
            metric_scores[metric].score * weight 
            for metric, weight in weights.items()
        )
        
        # Collect critical issues
        critical_issues = []
        for score in metric_scores.values():
            critical_issues.extend([
                issue for issue in score.issues 
                if issue.get('severity') == ValidationSeverity.CRITICAL.value
            ])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_scores, overall_score)
        
        report = DataQualityReport(
            provider_id=provider_id,
            provider_name=provider.name,
            assessment_date=datetime.utcnow(),
            date_range=(start_date, end_date),
            total_records=len(cost_data),
            overall_score=overall_score,
            metric_scores=metric_scores,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
        
        logger.info(
            "Data quality assessment completed",
            provider_id=str(provider_id),
            overall_score=overall_score,
            grade=report.get_grade(),
            total_records=len(cost_data),
            critical_issues=len(critical_issues)
        )
        
        return report
    
    async def _assess_completeness(self, cost_data: List[CostData]) -> QualityScore:
        """Assess data completeness"""
        total_records = len(cost_data)
        issues = []
        failed_count = 0
        
        required_fields = ['resource_id', 'resource_type', 'service_name', 'cost_amount', 'cost_date']
        
        for record in cost_data:
            record_issues = []
            
            # Check required fields
            for field in required_fields:
                value = getattr(record, field, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    record_issues.append({
                        'type': DataQualityIssueType.MISSING_REQUIRED_FIELD.value,
                        'severity': ValidationSeverity.ERROR.value,
                        'field': field,
                        'resource_id': record.resource_id,
                        'cost_date': record.cost_date.isoformat() if record.cost_date else None
                    })
            
            # Check optional but important fields
            if not record.tags or len(record.tags) == 0:
                record_issues.append({
                    'type': DataQualityIssueType.MISSING_TAGS.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'field': 'tags',
                    'resource_id': record.resource_id,
                    'cost_date': record.cost_date.isoformat() if record.cost_date else None
                })
            
            if record_issues:
                failed_count += 1
                issues.extend(record_issues)
        
        passed_count = total_records - failed_count
        score = passed_count / total_records if total_records > 0 else 0.0
        
        return QualityScore(
            metric=DataQualityMetric.COMPLETENESS,
            score=score,
            total_records=total_records,
            passed_records=passed_count,
            failed_records=failed_count,
            issues=issues
        )
    
    async def _assess_accuracy(self, cost_data: List[CostData]) -> QualityScore:
        """Assess data accuracy"""
        total_records = len(cost_data)
        issues = []
        failed_count = 0
        
        # Calculate cost statistics for outlier detection
        costs = [float(record.cost_amount) for record in cost_data if record.cost_amount is not None]
        
        if costs:
            mean_cost = statistics.mean(costs)
            std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
            outlier_threshold = mean_cost + (3 * std_cost)  # 3 standard deviations
        else:
            outlier_threshold = 0
        
        for record in cost_data:
            record_issues = []
            
            # Check for negative costs
            if record.cost_amount is not None and record.cost_amount < 0:
                record_issues.append({
                    'type': DataQualityIssueType.NEGATIVE_COST.value,
                    'severity': ValidationSeverity.ERROR.value,
                    'field': 'cost_amount',
                    'value': float(record.cost_amount),
                    'resource_id': record.resource_id,
                    'cost_date': record.cost_date.isoformat() if record.cost_date else None
                })
            
            # Check for cost outliers
            if (record.cost_amount is not None and 
                float(record.cost_amount) > outlier_threshold and 
                outlier_threshold > 0):
                record_issues.append({
                    'type': DataQualityIssueType.OUTLIER_COST.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'field': 'cost_amount',
                    'value': float(record.cost_amount),
                    'threshold': outlier_threshold,
                    'resource_id': record.resource_id,
                    'cost_date': record.cost_date.isoformat() if record.cost_date else None
                })
            
            # Check for future dates
            if record.cost_date and record.cost_date > date.today():
                record_issues.append({
                    'type': DataQualityIssueType.FUTURE_DATE.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'field': 'cost_date',
                    'value': record.cost_date.isoformat(),
                    'resource_id': record.resource_id
                })
            
            if record_issues:
                failed_count += 1
                issues.extend(record_issues)
        
        passed_count = total_records - failed_count
        score = passed_count / total_records if total_records > 0 else 0.0
        
        return QualityScore(
            metric=DataQualityMetric.ACCURACY,
            score=score,
            total_records=total_records,
            passed_records=passed_count,
            failed_records=failed_count,
            issues=issues
        )
    
    async def _assess_consistency(self, cost_data: List[CostData]) -> QualityScore:
        """Assess data consistency"""
        total_records = len(cost_data)
        issues = []
        failed_count = 0
        
        # Group by resource for consistency checks
        resource_groups = {}
        for record in cost_data:
            if record.resource_id:
                if record.resource_id not in resource_groups:
                    resource_groups[record.resource_id] = []
                resource_groups[record.resource_id].append(record)
        
        for resource_id, records in resource_groups.items():
            if len(records) < 2:
                continue
            
            # Check consistency of resource_type
            resource_types = set(r.resource_type for r in records if r.resource_type)
            if len(resource_types) > 1:
                failed_count += len(records)
                issues.append({
                    'type': DataQualityIssueType.INCONSISTENT_METADATA.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'field': 'resource_type',
                    'resource_id': resource_id,
                    'inconsistent_values': list(resource_types),
                    'message': f"Resource {resource_id} has inconsistent resource types"
                })
            
            # Check consistency of service_name
            service_names = set(r.service_name for r in records if r.service_name)
            if len(service_names) > 1:
                failed_count += len(records)
                issues.append({
                    'type': DataQualityIssueType.INCONSISTENT_METADATA.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'field': 'service_name',
                    'resource_id': resource_id,
                    'inconsistent_values': list(service_names),
                    'message': f"Resource {resource_id} has inconsistent service names"
                })
            
            # Check currency consistency
            currencies = set(r.currency for r in records if r.currency)
            if len(currencies) > 1:
                failed_count += len(records)
                issues.append({
                    'type': DataQualityIssueType.INCONSISTENT_METADATA.value,
                    'severity': ValidationSeverity.ERROR.value,
                    'field': 'currency',
                    'resource_id': resource_id,
                    'inconsistent_values': list(currencies),
                    'message': f"Resource {resource_id} has inconsistent currencies"
                })
        
        passed_count = total_records - failed_count
        score = passed_count / total_records if total_records > 0 else 0.0
        
        return QualityScore(
            metric=DataQualityMetric.CONSISTENCY,
            score=score,
            total_records=total_records,
            passed_records=passed_count,
            failed_records=failed_count,
            issues=issues
        )
    
    async def _assess_timeliness(self, cost_data: List[CostData], end_date: date) -> QualityScore:
        """Assess data timeliness"""
        total_records = len(cost_data)
        issues = []
        failed_count = 0
        
        # Check for data freshness (records should be recent)
        freshness_threshold = end_date - timedelta(days=3)  # Data should be within 3 days
        
        for record in cost_data:
            if record.cost_date and record.cost_date < freshness_threshold:
                # Check if this is expected historical data or stale data
                days_old = (end_date - record.cost_date).days
                if days_old > 7:  # More than a week old might be stale
                    failed_count += 1
                    issues.append({
                        'type': 'stale_data',
                        'severity': ValidationSeverity.WARNING.value,
                        'field': 'cost_date',
                        'value': record.cost_date.isoformat(),
                        'days_old': days_old,
                        'resource_id': record.resource_id,
                        'message': f"Data is {days_old} days old"
                    })
        
        passed_count = total_records - failed_count
        score = passed_count / total_records if total_records > 0 else 0.0
        
        return QualityScore(
            metric=DataQualityMetric.TIMELINESS,
            score=score,
            total_records=total_records,
            passed_records=passed_count,
            failed_records=failed_count,
            issues=issues
        )
    
    async def _assess_validity(self, cost_data: List[CostData]) -> QualityScore:
        """Assess data validity"""
        total_records = len(cost_data)
        issues = []
        failed_count = 0
        
        valid_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR'}
        
        for record in cost_data:
            record_issues = []
            
            # Validate currency
            if record.currency and record.currency not in valid_currencies:
                record_issues.append({
                    'type': DataQualityIssueType.INVALID_CURRENCY.value,
                    'severity': ValidationSeverity.ERROR.value,
                    'field': 'currency',
                    'value': record.currency,
                    'resource_id': record.resource_id,
                    'cost_date': record.cost_date.isoformat() if record.cost_date else None
                })
            
            # Validate resource_id format (should not be empty or just whitespace)
            if not record.resource_id or not record.resource_id.strip():
                record_issues.append({
                    'type': DataQualityIssueType.INVALID_DATA_TYPE.value,
                    'severity': ValidationSeverity.ERROR.value,
                    'field': 'resource_id',
                    'value': record.resource_id,
                    'message': "Resource ID is empty or invalid"
                })
            
            # Validate usage_quantity (should be non-negative if present)
            if record.usage_quantity is not None and record.usage_quantity < 0:
                record_issues.append({
                    'type': DataQualityIssueType.INVALID_DATA_TYPE.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'field': 'usage_quantity',
                    'value': float(record.usage_quantity),
                    'resource_id': record.resource_id,
                    'cost_date': record.cost_date.isoformat() if record.cost_date else None
                })
            
            if record_issues:
                failed_count += 1
                issues.extend(record_issues)
        
        passed_count = total_records - failed_count
        score = passed_count / total_records if total_records > 0 else 0.0
        
        return QualityScore(
            metric=DataQualityMetric.VALIDITY,
            score=score,
            total_records=total_records,
            passed_records=passed_count,
            failed_records=failed_count,
            issues=issues
        )
    
    async def _assess_uniqueness(self, cost_data: List[CostData]) -> QualityScore:
        """Assess data uniqueness"""
        total_records = len(cost_data)
        issues = []
        
        # Create unique key for each record
        record_keys = {}
        duplicate_count = 0
        
        for record in cost_data:
            # Create key based on fields that should be unique
            key = (
                record.provider_id,
                record.resource_id,
                record.cost_date,
                record.service_name,
                record.resource_type
            )
            
            if key in record_keys:
                duplicate_count += 1
                issues.append({
                    'type': DataQualityIssueType.DUPLICATE_RECORD.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'resource_id': record.resource_id,
                    'cost_date': record.cost_date.isoformat() if record.cost_date else None,
                    'service_name': record.service_name,
                    'message': "Duplicate record found"
                })
            else:
                record_keys[key] = record
        
        unique_count = total_records - duplicate_count
        score = unique_count / total_records if total_records > 0 else 0.0
        
        return QualityScore(
            metric=DataQualityMetric.UNIQUENESS,
            score=score,
            total_records=total_records,
            passed_records=unique_count,
            failed_records=duplicate_count,
            issues=issues
        )
    
    def _generate_recommendations(self, 
                                metric_scores: Dict[DataQualityMetric, QualityScore], 
                                overall_score: float) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Overall score recommendations
        if overall_score < 0.7:
            recommendations.append(
                "Overall data quality is below acceptable threshold. "
                "Immediate attention required to improve data collection processes."
            )
        elif overall_score < 0.85:
            recommendations.append(
                "Data quality has room for improvement. "
                "Consider implementing additional validation checks."
            )
        
        # Metric-specific recommendations
        for metric, score in metric_scores.items():
            if score.score < 0.8:
                if metric == DataQualityMetric.COMPLETENESS:
                    recommendations.append(
                        "Improve data completeness by ensuring all required fields are populated. "
                        "Review data collection processes and add validation at source."
                    )
                elif metric == DataQualityMetric.ACCURACY:
                    recommendations.append(
                        "Address data accuracy issues by implementing range checks and outlier detection. "
                        "Review cost calculation methods and data transformation logic."
                    )
                elif metric == DataQualityMetric.CONSISTENCY:
                    recommendations.append(
                        "Improve data consistency by standardizing resource naming and categorization. "
                        "Implement data normalization rules."
                    )
                elif metric == DataQualityMetric.TIMELINESS:
                    recommendations.append(
                        "Improve data timeliness by reducing data collection delays. "
                        "Consider more frequent synchronization schedules."
                    )
                elif metric == DataQualityMetric.VALIDITY:
                    recommendations.append(
                        "Enhance data validity by implementing stricter validation rules. "
                        "Review data formats and acceptable value ranges."
                    )
                elif metric == DataQualityMetric.UNIQUENESS:
                    recommendations.append(
                        "Address duplicate data by implementing deduplication logic. "
                        "Review data collection processes to prevent duplicate entries."
                    )
        
        # Add general recommendations if no specific issues found
        if not recommendations:
            recommendations.append(
                "Data quality is good. Continue monitoring and maintain current processes."
            )
        
        return recommendations