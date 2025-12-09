"""
Cost Anomaly Detection System for FinOps Platform
Detects unusual spending patterns and cost anomalies
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from dataclasses import dataclass
from enum import Enum
import statistics
import structlog

from .models import CostData, CloudProvider, Budget, BudgetAlert, AlertType
from .repositories import CostDataRepository, BudgetRepository, BudgetAlertRepository
from .cache_service import CacheService
from .logging_service import LoggingService

logger = structlog.get_logger(__name__)

class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnomalyType(Enum):
    """Types of cost anomalies"""
    COST_SPIKE = "cost_spike"
    COST_DROP = "cost_drop"
    USAGE_ANOMALY = "usage_anomaly"
    NEW_RESOURCE = "new_resource"
    MISSING_RESOURCE = "missing_resource"
    BUDGET_THRESHOLD = "budget_threshold"
    TREND_DEVIATION = "trend_deviation"
    SEASONAL_ANOMALY = "seasonal_anomaly"

@dataclass
class CostAnomaly:
    """Represents a detected cost anomaly"""
    id: str
    provider_id: UUID
    resource_id: Optional[str]
    service_name: Optional[str]
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    detected_at: datetime
    cost_date: date
    current_cost: Decimal
    expected_cost: Decimal
    deviation_percentage: float
    confidence_score: float  # 0.0 to 1.0
    description: str
    impact_assessment: str
    recommended_actions: List[str]
    context: Dict[str, Any]

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    resource_id: str
    service_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    daily_change_rate: float
    weekly_change_rate: float
    monthly_change_rate: float
    prediction_7_days: Decimal
    prediction_30_days: Decimal

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection process"""
    provider_id: UUID
    detection_date: datetime
    total_anomalies: int
    anomalies_by_severity: Dict[AnomalySeverity, int]
    anomalies_by_type: Dict[AnomalyType, int]
    anomalies: List[CostAnomaly]
    trend_analyses: List[TrendAnalysis]
    summary: Dict[str, Any]

class CostAnomalyDetector:
    """Advanced cost anomaly detection system"""
    
    def __init__(self, 
                 cost_data_repository: CostDataRepository,
                 budget_repository: BudgetRepository,
                 budget_alert_repository: BudgetAlertRepository,
                 cache_service: CacheService,
                 logging_service: LoggingService):
        self.cost_data_repo = cost_data_repository
        self.budget_repo = budget_repository
        self.budget_alert_repo = budget_alert_repository
        self.cache_service = cache_service
        self.logging_service = logging_service
        
        # Configuration parameters
        self.spike_threshold = 2.0  # 200% increase
        self.drop_threshold = 0.5   # 50% decrease
        self.trend_window_days = 14  # Days for trend analysis
        self.seasonal_window_weeks = 4  # Weeks for seasonal comparison
        self.confidence_threshold = 0.7  # Minimum confidence for reporting
    
    async def detect_anomalies(self, 
                             provider_id: UUID, 
                             detection_date: date = None,
                             lookback_days: int = 30) -> AnomalyDetectionResult:
        """
        Detect cost anomalies for a provider
        
        Args:
            provider_id: UUID of the cloud provider
            detection_date: Date to run detection for (defaults to today)
            lookback_days: Number of days to analyze
            
        Returns:
            AnomalyDetectionResult with detected anomalies
        """
        if not detection_date:
            detection_date = date.today()
        
        start_date = detection_date - timedelta(days=lookback_days)
        
        self.logging_service.info(
            "Starting cost anomaly detection",
            provider_id=str(provider_id),
            detection_date=detection_date.isoformat(),
            lookback_days=lookback_days
        )
        
        # Get cost data for analysis
        cost_data = await self.cost_data_repo.get_cost_data_by_date_range(
            provider_id, start_date, detection_date
        )
        
        if not cost_data:
            self.logging_service.warning(
                "No cost data available for anomaly detection",
                provider_id=str(provider_id)
            )
            return AnomalyDetectionResult(
                provider_id=provider_id,
                detection_date=datetime.utcnow(),
                total_anomalies=0,
                anomalies_by_severity={},
                anomalies_by_type={},
                anomalies=[],
                trend_analyses=[],
                summary={"message": "No data available"}
            )
        
        # Detect different types of anomalies
        anomalies = []
        
        # 1. Cost spike detection
        spike_anomalies = await self._detect_cost_spikes(cost_data, detection_date)
        anomalies.extend(spike_anomalies)
        
        # 2. Cost drop detection
        drop_anomalies = await self._detect_cost_drops(cost_data, detection_date)
        anomalies.extend(drop_anomalies)
        
        # 3. New resource detection
        new_resource_anomalies = await self._detect_new_resources(cost_data, detection_date)
        anomalies.extend(new_resource_anomalies)
        
        # 4. Missing resource detection
        missing_resource_anomalies = await self._detect_missing_resources(cost_data, detection_date)
        anomalies.extend(missing_resource_anomalies)
        
        # 5. Budget threshold anomalies
        budget_anomalies = await self._detect_budget_anomalies(provider_id, detection_date)
        anomalies.extend(budget_anomalies)
        
        # 6. Trend deviation detection
        trend_anomalies = await self._detect_trend_deviations(cost_data, detection_date)
        anomalies.extend(trend_anomalies)
        
        # 7. Seasonal anomalies
        seasonal_anomalies = await self._detect_seasonal_anomalies(cost_data, detection_date)
        anomalies.extend(seasonal_anomalies)
        
        # Generate trend analyses
        trend_analyses = await self._analyze_trends(cost_data, detection_date)
        
        # Filter anomalies by confidence threshold
        filtered_anomalies = [
            anomaly for anomaly in anomalies 
            if anomaly.confidence_score >= self.confidence_threshold
        ]
        
        # Calculate statistics
        anomalies_by_severity = {}
        anomalies_by_type = {}
        
        for anomaly in filtered_anomalies:
            # Count by severity
            if anomaly.severity not in anomalies_by_severity:
                anomalies_by_severity[anomaly.severity] = 0
            anomalies_by_severity[anomaly.severity] += 1
            
            # Count by type
            if anomaly.anomaly_type not in anomalies_by_type:
                anomalies_by_type[anomaly.anomaly_type] = 0
            anomalies_by_type[anomaly.anomaly_type] += 1
        
        # Create summary
        total_cost = sum(float(record.cost_amount) for record in cost_data)
        anomalous_cost = sum(
            float(anomaly.current_cost) for anomaly in filtered_anomalies
            if anomaly.current_cost
        )
        
        summary = {
            "total_cost_analyzed": total_cost,
            "anomalous_cost": anomalous_cost,
            "anomalous_cost_percentage": (anomalous_cost / total_cost * 100) if total_cost > 0 else 0,
            "detection_period_days": lookback_days,
            "resources_analyzed": len(set(record.resource_id for record in cost_data)),
            "services_analyzed": len(set(record.service_name for record in cost_data))
        }
        
        result = AnomalyDetectionResult(
            provider_id=provider_id,
            detection_date=datetime.utcnow(),
            total_anomalies=len(filtered_anomalies),
            anomalies_by_severity=anomalies_by_severity,
            anomalies_by_type=anomalies_by_type,
            anomalies=filtered_anomalies,
            trend_analyses=trend_analyses,
            summary=summary
        )
        
        self.logging_service.info(
            "Cost anomaly detection completed",
            provider_id=str(provider_id),
            total_anomalies=len(filtered_anomalies),
            critical_anomalies=anomalies_by_severity.get(AnomalySeverity.CRITICAL, 0),
            high_anomalies=anomalies_by_severity.get(AnomalySeverity.HIGH, 0)
        )
        
        # Cache results
        await self._cache_detection_results(provider_id, detection_date, result)
        
        return result
    
    async def _detect_cost_spikes(self, cost_data: List[CostData], detection_date: date) -> List[CostAnomaly]:
        """Detect cost spikes (sudden increases in spending)"""
        anomalies = []
        
        # Group by resource and service
        resource_costs = {}
        for record in cost_data:
            key = (record.resource_id, record.service_name)
            if key not in resource_costs:
                resource_costs[key] = []
            resource_costs[key].append(record)
        
        for (resource_id, service_name), records in resource_costs.items():
            if len(records) < 3:  # Need at least 3 data points
                continue
            
            # Sort by date
            records.sort(key=lambda x: x.cost_date)
            
            # Get recent costs (last 3 days)
            recent_records = [r for r in records if r.cost_date >= detection_date - timedelta(days=3)]
            historical_records = [r for r in records if r.cost_date < detection_date - timedelta(days=3)]
            
            if not recent_records or not historical_records:
                continue
            
            # Calculate averages
            recent_avg = statistics.mean(float(r.cost_amount) for r in recent_records)
            historical_avg = statistics.mean(float(r.cost_amount) for r in historical_records)
            
            if historical_avg == 0:
                continue
            
            # Check for spike
            spike_ratio = recent_avg / historical_avg
            if spike_ratio >= self.spike_threshold:
                deviation_percentage = (spike_ratio - 1) * 100
                
                # Calculate confidence based on consistency and magnitude
                confidence = min(1.0, (spike_ratio - 1) / 2)  # Higher spikes = higher confidence
                
                # Determine severity
                if spike_ratio >= 5.0:
                    severity = AnomalySeverity.CRITICAL
                elif spike_ratio >= 3.0:
                    severity = AnomalySeverity.HIGH
                elif spike_ratio >= 2.0:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW
                
                anomaly = CostAnomaly(
                    id=f"spike_{resource_id}_{service_name}_{detection_date.isoformat()}",
                    provider_id=records[0].provider_id,
                    resource_id=resource_id,
                    service_name=service_name,
                    anomaly_type=AnomalyType.COST_SPIKE,
                    severity=severity,
                    detected_at=datetime.utcnow(),
                    cost_date=detection_date,
                    current_cost=Decimal(str(recent_avg)),
                    expected_cost=Decimal(str(historical_avg)),
                    deviation_percentage=deviation_percentage,
                    confidence_score=confidence,
                    description=f"Cost spike detected: {spike_ratio:.1f}x increase in spending",
                    impact_assessment=f"Additional ${recent_avg - historical_avg:.2f} per day",
                    recommended_actions=[
                        "Investigate recent changes to the resource",
                        "Check for increased usage or new workloads",
                        "Review resource configuration changes",
                        "Consider cost optimization opportunities"
                    ],
                    context={
                        "spike_ratio": spike_ratio,
                        "recent_avg_cost": recent_avg,
                        "historical_avg_cost": historical_avg,
                        "data_points": len(records)
                    }
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_cost_drops(self, cost_data: List[CostData], detection_date: date) -> List[CostAnomaly]:
        """Detect cost drops (sudden decreases in spending)"""
        anomalies = []
        
        # Group by resource and service
        resource_costs = {}
        for record in cost_data:
            key = (record.resource_id, record.service_name)
            if key not in resource_costs:
                resource_costs[key] = []
            resource_costs[key].append(record)
        
        for (resource_id, service_name), records in resource_costs.items():
            if len(records) < 3:
                continue
            
            records.sort(key=lambda x: x.cost_date)
            
            # Get recent costs (last 3 days)
            recent_records = [r for r in records if r.cost_date >= detection_date - timedelta(days=3)]
            historical_records = [r for r in records if r.cost_date < detection_date - timedelta(days=3)]
            
            if not recent_records or not historical_records:
                continue
            
            recent_avg = statistics.mean(float(r.cost_amount) for r in recent_records)
            historical_avg = statistics.mean(float(r.cost_amount) for r in historical_records)
            
            if historical_avg == 0:
                continue
            
            # Check for drop
            drop_ratio = recent_avg / historical_avg
            if drop_ratio <= self.drop_threshold:
                deviation_percentage = (1 - drop_ratio) * 100
                
                # Calculate confidence
                confidence = min(1.0, (1 - drop_ratio) / 0.5)
                
                # Determine severity (drops are generally less severe than spikes)
                if drop_ratio <= 0.1:
                    severity = AnomalySeverity.HIGH
                elif drop_ratio <= 0.3:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW
                
                anomaly = CostAnomaly(
                    id=f"drop_{resource_id}_{service_name}_{detection_date.isoformat()}",
                    provider_id=records[0].provider_id,
                    resource_id=resource_id,
                    service_name=service_name,
                    anomaly_type=AnomalyType.COST_DROP,
                    severity=severity,
                    detected_at=datetime.utcnow(),
                    cost_date=detection_date,
                    current_cost=Decimal(str(recent_avg)),
                    expected_cost=Decimal(str(historical_avg)),
                    deviation_percentage=deviation_percentage,
                    confidence_score=confidence,
                    description=f"Cost drop detected: {(1-drop_ratio)*100:.1f}% decrease in spending",
                    impact_assessment=f"Savings of ${historical_avg - recent_avg:.2f} per day",
                    recommended_actions=[
                        "Verify if the cost reduction is intentional",
                        "Check if resources were terminated or scaled down",
                        "Ensure services are still functioning properly",
                        "Document cost optimization if intentional"
                    ],
                    context={
                        "drop_ratio": drop_ratio,
                        "recent_avg_cost": recent_avg,
                        "historical_avg_cost": historical_avg,
                        "data_points": len(records)
                    }
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_new_resources(self, cost_data: List[CostData], detection_date: date) -> List[CostAnomaly]:
        """Detect new resources that started incurring costs"""
        anomalies = []
        
        # Find resources that first appeared in the last few days
        resource_first_seen = {}
        for record in cost_data:
            if record.resource_id not in resource_first_seen:
                resource_first_seen[record.resource_id] = record.cost_date
            else:
                resource_first_seen[record.resource_id] = min(
                    resource_first_seen[record.resource_id], 
                    record.cost_date
                )
        
        # Check for resources that are new (first seen in last 7 days)
        new_threshold = detection_date - timedelta(days=7)
        
        for resource_id, first_seen in resource_first_seen.items():
            if first_seen >= new_threshold:
                # Get cost data for this resource
                resource_records = [r for r in cost_data if r.resource_id == resource_id]
                total_cost = sum(float(r.cost_amount) for r in resource_records)
                
                if total_cost > 10:  # Only flag if cost is significant
                    severity = AnomalySeverity.MEDIUM if total_cost > 100 else AnomalySeverity.LOW
                    
                    anomaly = CostAnomaly(
                        id=f"new_resource_{resource_id}_{detection_date.isoformat()}",
                        provider_id=resource_records[0].provider_id,
                        resource_id=resource_id,
                        service_name=resource_records[0].service_name,
                        anomaly_type=AnomalyType.NEW_RESOURCE,
                        severity=severity,
                        detected_at=datetime.utcnow(),
                        cost_date=detection_date,
                        current_cost=Decimal(str(total_cost)),
                        expected_cost=Decimal('0'),
                        deviation_percentage=100.0,
                        confidence_score=0.9,
                        description=f"New resource detected with ${total_cost:.2f} in costs",
                        impact_assessment=f"New spending of ${total_cost:.2f}",
                        recommended_actions=[
                            "Verify if this resource was intentionally created",
                            "Check if proper approvals were obtained",
                            "Review resource configuration and sizing",
                            "Add appropriate cost allocation tags"
                        ],
                        context={
                            "first_seen_date": first_seen.isoformat(),
                            "days_since_creation": (detection_date - first_seen).days,
                            "total_cost": total_cost
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_missing_resources(self, cost_data: List[CostData], detection_date: date) -> List[CostAnomaly]:
        """Detect resources that stopped incurring costs"""
        anomalies = []
        
        # Find resources that haven't had costs in the last few days
        resource_last_seen = {}
        for record in cost_data:
            if record.resource_id not in resource_last_seen:
                resource_last_seen[record.resource_id] = record.cost_date
            else:
                resource_last_seen[record.resource_id] = max(
                    resource_last_seen[record.resource_id], 
                    record.cost_date
                )
        
        # Check for resources that haven't been seen recently
        missing_threshold = detection_date - timedelta(days=3)
        
        for resource_id, last_seen in resource_last_seen.items():
            if last_seen < missing_threshold:
                # Get historical cost for this resource
                resource_records = [r for r in cost_data if r.resource_id == resource_id]
                avg_cost = statistics.mean(float(r.cost_amount) for r in resource_records)
                
                if avg_cost > 5:  # Only flag if it was a significant cost
                    days_missing = (detection_date - last_seen).days
                    severity = AnomalySeverity.MEDIUM if avg_cost > 50 else AnomalySeverity.LOW
                    
                    anomaly = CostAnomaly(
                        id=f"missing_resource_{resource_id}_{detection_date.isoformat()}",
                        provider_id=resource_records[0].provider_id,
                        resource_id=resource_id,
                        service_name=resource_records[0].service_name,
                        anomaly_type=AnomalyType.MISSING_RESOURCE,
                        severity=severity,
                        detected_at=datetime.utcnow(),
                        cost_date=detection_date,
                        current_cost=Decimal('0'),
                        expected_cost=Decimal(str(avg_cost)),
                        deviation_percentage=100.0,
                        confidence_score=0.8,
                        description=f"Resource missing for {days_missing} days (was ${avg_cost:.2f}/day)",
                        impact_assessment=f"Potential savings of ${avg_cost:.2f} per day",
                        recommended_actions=[
                            "Verify if resource was intentionally terminated",
                            "Check if resource was moved or renamed",
                            "Ensure no unexpected service interruptions",
                            "Update cost allocation if resource was decommissioned"
                        ],
                        context={
                            "last_seen_date": last_seen.isoformat(),
                            "days_missing": days_missing,
                            "historical_avg_cost": avg_cost
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_budget_anomalies(self, provider_id: UUID, detection_date: date) -> List[CostAnomaly]:
        """Detect budget threshold breaches"""
        anomalies = []
        
        # Get active budgets for this provider
        budgets = await self.budget_repo.get_budgets_for_date(detection_date)
        provider_budgets = [b for b in budgets if self._budget_applies_to_provider(b, provider_id)]
        
        for budget in provider_budgets:
            # Calculate current spend for this budget
            current_spend = await self._calculate_budget_spend(budget, detection_date)
            spend_percentage = (current_spend / budget.amount) * 100 if budget.amount > 0 else 0
            
            # Check if any thresholds are breached
            for threshold in budget.alert_thresholds:
                if spend_percentage >= threshold:
                    # Check if we already have an alert for this threshold
                    existing_alerts = await self.budget_alert_repo.get_alerts_by_budget(budget.id)
                    threshold_alert_exists = any(
                        alert.threshold_percentage == threshold and not alert.acknowledged
                        for alert in existing_alerts
                    )
                    
                    if not threshold_alert_exists:
                        severity = self._calculate_budget_severity(spend_percentage, threshold)
                        
                        anomaly = CostAnomaly(
                            id=f"budget_threshold_{budget.id}_{threshold}_{detection_date.isoformat()}",
                            provider_id=provider_id,
                            resource_id=None,
                            service_name=None,
                            anomaly_type=AnomalyType.BUDGET_THRESHOLD,
                            severity=severity,
                            detected_at=datetime.utcnow(),
                            cost_date=detection_date,
                            current_cost=current_spend,
                            expected_cost=budget.amount * Decimal(str(threshold / 100)),
                            deviation_percentage=spend_percentage - threshold,
                            confidence_score=1.0,
                            description=f"Budget '{budget.name}' exceeded {threshold}% threshold ({spend_percentage:.1f}%)",
                            impact_assessment=f"Over budget by ${current_spend - (budget.amount * Decimal(str(threshold / 100))):.2f}",
                            recommended_actions=[
                                "Review spending patterns and identify cost drivers",
                                "Consider implementing cost controls",
                                "Evaluate if budget needs adjustment",
                                "Notify stakeholders about budget breach"
                            ],
                            context={
                                "budget_id": str(budget.id),
                                "budget_name": budget.name,
                                "budget_amount": float(budget.amount),
                                "threshold_percentage": threshold,
                                "current_spend": float(current_spend),
                                "spend_percentage": spend_percentage
                            }
                        )
                        
                        anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_trend_deviations(self, cost_data: List[CostData], detection_date: date) -> List[CostAnomaly]:
        """Detect deviations from expected cost trends"""
        anomalies = []
        
        # Group by service for trend analysis
        service_costs = {}
        for record in cost_data:
            if record.service_name not in service_costs:
                service_costs[record.service_name] = []
            service_costs[record.service_name].append(record)
        
        for service_name, records in service_costs.items():
            if len(records) < 7:  # Need at least a week of data
                continue
            
            # Sort by date and calculate daily totals
            records.sort(key=lambda x: x.cost_date)
            daily_costs = {}
            
            for record in records:
                if record.cost_date not in daily_costs:
                    daily_costs[record.cost_date] = 0
                daily_costs[record.cost_date] += float(record.cost_amount)
            
            # Calculate trend
            dates = sorted(daily_costs.keys())
            costs = [daily_costs[d] for d in dates]
            
            if len(costs) >= 7:
                # Simple linear trend calculation
                trend_slope = self._calculate_trend_slope(costs)
                recent_cost = costs[-1]
                expected_cost = costs[-2] + trend_slope if len(costs) > 1 else recent_cost
                
                deviation = abs(recent_cost - expected_cost) / expected_cost if expected_cost > 0 else 0
                
                if deviation > 0.3:  # 30% deviation from trend
                    severity = AnomalySeverity.MEDIUM if deviation > 0.5 else AnomalySeverity.LOW
                    
                    anomaly = CostAnomaly(
                        id=f"trend_deviation_{service_name}_{detection_date.isoformat()}",
                        provider_id=records[0].provider_id,
                        resource_id=None,
                        service_name=service_name,
                        anomaly_type=AnomalyType.TREND_DEVIATION,
                        severity=severity,
                        detected_at=datetime.utcnow(),
                        cost_date=detection_date,
                        current_cost=Decimal(str(recent_cost)),
                        expected_cost=Decimal(str(expected_cost)),
                        deviation_percentage=deviation * 100,
                        confidence_score=0.7,
                        description=f"Service cost deviated {deviation*100:.1f}% from expected trend",
                        impact_assessment=f"Unexpected cost variance of ${abs(recent_cost - expected_cost):.2f}",
                        recommended_actions=[
                            "Analyze recent changes in service usage",
                            "Review service configuration changes",
                            "Check for seasonal or cyclical patterns",
                            "Investigate underlying cost drivers"
                        ],
                        context={
                            "trend_slope": trend_slope,
                            "recent_cost": recent_cost,
                            "expected_cost": expected_cost,
                            "data_points": len(costs)
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_seasonal_anomalies(self, cost_data: List[CostData], detection_date: date) -> List[CostAnomaly]:
        """Detect seasonal anomalies by comparing to same day of week in previous weeks"""
        anomalies = []
        
        # Get the day of week for detection date
        target_weekday = detection_date.weekday()
        
        # Group costs by service and day
        service_daily_costs = {}
        for record in cost_data:
            if record.service_name not in service_daily_costs:
                service_daily_costs[record.service_name] = {}
            
            if record.cost_date not in service_daily_costs[record.service_name]:
                service_daily_costs[record.service_name][record.cost_date] = 0
            
            service_daily_costs[record.service_name][record.cost_date] += float(record.cost_amount)
        
        for service_name, daily_costs in service_daily_costs.items():
            # Get costs for the same weekday in previous weeks
            same_weekday_costs = []
            current_cost = daily_costs.get(detection_date, 0)
            
            for cost_date, cost in daily_costs.items():
                if (cost_date.weekday() == target_weekday and 
                    cost_date != detection_date and 
                    cost_date >= detection_date - timedelta(weeks=self.seasonal_window_weeks)):
                    same_weekday_costs.append(cost)
            
            if len(same_weekday_costs) >= 2 and current_cost > 0:
                expected_cost = statistics.mean(same_weekday_costs)
                deviation = abs(current_cost - expected_cost) / expected_cost if expected_cost > 0 else 0
                
                if deviation > 0.4:  # 40% deviation from seasonal pattern
                    severity = AnomalySeverity.MEDIUM if deviation > 0.7 else AnomalySeverity.LOW
                    
                    anomaly = CostAnomaly(
                        id=f"seasonal_{service_name}_{detection_date.isoformat()}",
                        provider_id=cost_data[0].provider_id,
                        resource_id=None,
                        service_name=service_name,
                        anomaly_type=AnomalyType.SEASONAL_ANOMALY,
                        severity=severity,
                        detected_at=datetime.utcnow(),
                        cost_date=detection_date,
                        current_cost=Decimal(str(current_cost)),
                        expected_cost=Decimal(str(expected_cost)),
                        deviation_percentage=deviation * 100,
                        confidence_score=0.6,
                        description=f"Seasonal anomaly: {deviation*100:.1f}% deviation from typical {detection_date.strftime('%A')} costs",
                        impact_assessment=f"Variance of ${abs(current_cost - expected_cost):.2f} from seasonal pattern",
                        recommended_actions=[
                            "Compare to same period in previous months/years",
                            "Check for seasonal business changes",
                            "Review holiday or event impacts",
                            "Analyze weekly usage patterns"
                        ],
                        context={
                            "weekday": detection_date.strftime('%A'),
                            "historical_same_weekday_costs": same_weekday_costs,
                            "seasonal_window_weeks": self.seasonal_window_weeks
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _analyze_trends(self, cost_data: List[CostData], detection_date: date) -> List[TrendAnalysis]:
        """Analyze cost trends for resources and services"""
        trend_analyses = []
        
        # Group by resource and service
        resource_service_costs = {}
        for record in cost_data:
            key = (record.resource_id, record.service_name)
            if key not in resource_service_costs:
                resource_service_costs[key] = []
            resource_service_costs[key].append(record)
        
        for (resource_id, service_name), records in resource_service_costs.items():
            if len(records) < 7:  # Need at least a week of data
                continue
            
            # Sort by date and calculate daily totals
            records.sort(key=lambda x: x.cost_date)
            daily_costs = {}
            
            for record in records:
                if record.cost_date not in daily_costs:
                    daily_costs[record.cost_date] = 0
                daily_costs[record.cost_date] += float(record.cost_amount)
            
            dates = sorted(daily_costs.keys())
            costs = [daily_costs[d] for d in dates]
            
            # Calculate trends
            daily_change_rate = self._calculate_change_rate(costs, 1)
            weekly_change_rate = self._calculate_change_rate(costs, 7)
            monthly_change_rate = self._calculate_change_rate(costs, 30)
            
            # Determine trend direction and strength
            trend_slope = self._calculate_trend_slope(costs)
            trend_direction = "increasing" if trend_slope > 0.1 else "decreasing" if trend_slope < -0.1 else "stable"
            trend_strength = min(1.0, abs(trend_slope) / statistics.mean(costs) if statistics.mean(costs) > 0 else 0)
            
            # Make predictions
            current_cost = costs[-1]
            prediction_7_days = max(0, current_cost + (trend_slope * 7))
            prediction_30_days = max(0, current_cost + (trend_slope * 30))
            
            trend_analysis = TrendAnalysis(
                resource_id=resource_id,
                service_name=service_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                daily_change_rate=daily_change_rate,
                weekly_change_rate=weekly_change_rate,
                monthly_change_rate=monthly_change_rate,
                prediction_7_days=Decimal(str(prediction_7_days)),
                prediction_30_days=Decimal(str(prediction_30_days))
            )
            
            trend_analyses.append(trend_analysis)
        
        return trend_analyses
    
    def _calculate_trend_slope(self, costs: List[float]) -> float:
        """Calculate the slope of cost trend using simple linear regression"""
        if len(costs) < 2:
            return 0.0
        
        n = len(costs)
        x_values = list(range(n))
        
        # Calculate slope using least squares method
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(costs)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, costs))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_change_rate(self, costs: List[float], period: int) -> float:
        """Calculate the rate of change over a specific period"""
        if len(costs) < period + 1:
            return 0.0
        
        recent_avg = statistics.mean(costs[-period:])
        previous_avg = statistics.mean(costs[-2*period:-period]) if len(costs) >= 2*period else costs[0]
        
        return (recent_avg - previous_avg) / previous_avg if previous_avg != 0 else 0.0
    
    def _budget_applies_to_provider(self, budget: Budget, provider_id: UUID) -> bool:
        """Check if a budget applies to a specific provider"""
        # This is a simplified check - in practice, you'd check the budget's scope_filters
        # to see if it includes this provider
        return True  # For now, assume all budgets apply to all providers
    
    async def _calculate_budget_spend(self, budget: Budget, target_date: date) -> Decimal:
        """Calculate current spend for a budget"""
        # This is a simplified implementation
        # In practice, you'd apply the budget's scope_filters to calculate actual spend
        return Decimal('1000.00')  # Placeholder
    
    def _calculate_budget_severity(self, spend_percentage: float, threshold: float) -> AnomalySeverity:
        """Calculate severity based on budget threshold breach"""
        if spend_percentage >= threshold + 50:  # 50% over threshold
            return AnomalySeverity.CRITICAL
        elif spend_percentage >= threshold + 25:  # 25% over threshold
            return AnomalySeverity.HIGH
        elif spend_percentage >= threshold + 10:  # 10% over threshold
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    async def _cache_detection_results(self, provider_id: UUID, detection_date: date, result: AnomalyDetectionResult) -> None:
        """Cache anomaly detection results"""
        try:
            cache_key = f"anomaly_detection:{provider_id}:{detection_date.isoformat()}"
            
            # Create cacheable summary
            cache_data = {
                "total_anomalies": result.total_anomalies,
                "anomalies_by_severity": {k.value: v for k, v in result.anomalies_by_severity.items()},
                "anomalies_by_type": {k.value: v for k, v in result.anomalies_by_type.items()},
                "summary": result.summary,
                "detection_date": result.detection_date.isoformat()
            }
            
            await self.cache_service.set(cache_key, cache_data, ttl=86400)  # 24 hours
        except Exception as e:
            self.logging_service.warning(
                "Failed to cache anomaly detection results",
                provider_id=str(provider_id),
                error=str(e)
            )