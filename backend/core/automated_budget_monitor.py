"""
Automated Budget Monitoring System for FinOps Platform
Provides real-time budget tracking, forecasting, and automated alerting
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

from .models import Budget, BudgetAlert, CostData, AlertType, BudgetType
from .repositories import BudgetRepository, BudgetAlertRepository, CostDataRepository
from .cache_service import CacheService
from .logging_service import LoggingService
from .alert_manager import AlertManager

logger = structlog.get_logger(__name__)

class ForecastMethod(Enum):
    """Budget forecasting methods"""
    LINEAR_TREND = "linear_trend"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"

class BudgetHealthStatus(Enum):
    """Budget health status levels"""
    HEALTHY = "healthy"
    AT_RISK = "at_risk"
    OVER_BUDGET = "over_budget"
    CRITICAL = "critical"

@dataclass
class BudgetUtilization:
    """Budget utilization metrics"""
    budget_id: UUID
    budget_name: str
    budget_amount: Decimal
    current_spend: Decimal
    utilization_percentage: float
    remaining_amount: Decimal
    days_elapsed: int
    days_remaining: int
    daily_burn_rate: Decimal
    projected_spend: Decimal
    projected_overage: Decimal
    health_status: BudgetHealthStatus
    last_updated: datetime

@dataclass
class BudgetForecast:
    """Budget spending forecast"""
    budget_id: UUID
    forecast_date: datetime
    method: ForecastMethod
    confidence_level: float
    forecasted_spend_7_days: Decimal
    forecasted_spend_30_days: Decimal
    forecasted_spend_end_of_period: Decimal
    probability_of_overage: float
    recommended_actions: List[str]
    forecast_accuracy: Optional[float] = None

@dataclass
class BudgetMonitoringResult:
    """Result of budget monitoring process"""
    monitoring_date: datetime
    total_budgets_monitored: int
    budgets_at_risk: int
    budgets_over_budget: int
    total_alerts_generated: int
    utilizations: List[BudgetUtilization]
    forecasts: List[BudgetForecast]
    summary: Dict[str, Any]

class AutomatedBudgetMonitor:
    """Automated budget monitoring with real-time tracking and forecasting"""
    
    def __init__(self,
                 budget_repository: BudgetRepository,
                 budget_alert_repository: BudgetAlertRepository,
                 cost_data_repository: CostDataRepository,
                 cache_service: CacheService,
                 logging_service: LoggingService,
                 alert_manager: AlertManager):
        self.budget_repo = budget_repository
        self.budget_alert_repo = budget_alert_repository
        self.cost_data_repo = cost_data_repository
        self.cache_service = cache_service
        self.logging_service = logging_service
        self.alert_manager = alert_manager
        
        # Configuration
        self.monitoring_interval_minutes = 60  # Monitor every hour
        self.forecast_confidence_threshold = 0.7
        self.at_risk_threshold = 0.8  # 80% utilization
        self.critical_threshold = 1.0  # 100% utilization
        
        # Monitoring state
        self.is_monitoring = False
        self.last_monitoring_run = None
    
    async def start_monitoring(self) -> None:
        """Start automated budget monitoring"""
        if self.is_monitoring:
            self.logging_service.warning("Budget monitoring is already running")
            return
        
        self.is_monitoring = True
        self.logging_service.info("Starting automated budget monitoring")
        
        try:
            while self.is_monitoring:
                await self.run_monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval_minutes * 60)
        except Exception as e:
            self.logging_service.error("Budget monitoring stopped due to error", error=str(e))
            self.is_monitoring = False
    
    async def stop_monitoring(self) -> None:
        """Stop automated budget monitoring"""
        self.is_monitoring = False
        self.logging_service.info("Stopped automated budget monitoring")
    
    async def run_monitoring_cycle(self) -> BudgetMonitoringResult:
        """Run a single monitoring cycle"""
        monitoring_start = datetime.utcnow()
        
        self.logging_service.info("Starting budget monitoring cycle")
        
        try:
            # Get all active budgets
            active_budgets = await self.budget_repo.get_active_budgets()
            
            if not active_budgets:
                self.logging_service.info("No active budgets found for monitoring")
                return BudgetMonitoringResult(
                    monitoring_date=monitoring_start,
                    total_budgets_monitored=0,
                    budgets_at_risk=0,
                    budgets_over_budget=0,
                    total_alerts_generated=0,
                    utilizations=[],
                    forecasts=[],
                    summary={"message": "No active budgets"}
                )
            
            # Monitor each budget
            utilizations = []
            forecasts = []
            alerts_generated = 0
            budgets_at_risk = 0
            budgets_over_budget = 0
            
            for budget in active_budgets:
                try:
                    # Calculate budget utilization
                    utilization = await self.calculate_budget_utilization(budget)
                    utilizations.append(utilization)
                    
                    # Generate forecast
                    forecast = await self.generate_budget_forecast(budget)
                    forecasts.append(forecast)
                    
                    # Check for alerts
                    new_alerts = await self.check_budget_alerts(budget, utilization, forecast)
                    alerts_generated += len(new_alerts)
                    
                    # Update counters
                    if utilization.health_status == BudgetHealthStatus.AT_RISK:
                        budgets_at_risk += 1
                    elif utilization.health_status in [BudgetHealthStatus.OVER_BUDGET, BudgetHealthStatus.CRITICAL]:
                        budgets_over_budget += 1
                    
                    # Cache utilization data
                    await self._cache_budget_utilization(budget.id, utilization)
                    
                except Exception as e:
                    self.logging_service.error(
                        "Error monitoring budget",
                        budget_id=str(budget.id),
                        budget_name=budget.name,
                        error=str(e)
                    )
            
            # Create summary
            summary = {
                "monitoring_duration_seconds": (datetime.utcnow() - monitoring_start).total_seconds(),
                "average_utilization": statistics.mean([u.utilization_percentage for u in utilizations]) if utilizations else 0,
                "total_budget_amount": sum(float(u.budget_amount) for u in utilizations),
                "total_current_spend": sum(float(u.current_spend) for u in utilizations),
                "total_projected_overage": sum(float(u.projected_overage) for u in utilizations),
                "high_confidence_forecasts": len([f for f in forecasts if f.confidence_level >= self.forecast_confidence_threshold])
            }
            
            result = BudgetMonitoringResult(
                monitoring_date=monitoring_start,
                total_budgets_monitored=len(active_budgets),
                budgets_at_risk=budgets_at_risk,
                budgets_over_budget=budgets_over_budget,
                total_alerts_generated=alerts_generated,
                utilizations=utilizations,
                forecasts=forecasts,
                summary=summary
            )
            
            self.last_monitoring_run = monitoring_start
            
            self.logging_service.info(
                "Budget monitoring cycle completed",
                total_budgets=len(active_budgets),
                budgets_at_risk=budgets_at_risk,
                budgets_over_budget=budgets_over_budget,
                alerts_generated=alerts_generated
            )
            
            return result
            
        except Exception as e:
            self.logging_service.error("Budget monitoring cycle failed", error=str(e))
            raise
    
    async def calculate_budget_utilization(self, budget: Budget) -> BudgetUtilization:
        """Calculate current budget utilization metrics"""
        try:
            # Get current spending for this budget
            current_spend = await self._calculate_current_spend(budget)
            
            # Calculate utilization percentage
            utilization_percentage = (current_spend / budget.amount) * 100 if budget.amount > 0 else 0
            
            # Calculate remaining amount
            remaining_amount = budget.amount - current_spend
            
            # Calculate time metrics
            now = datetime.utcnow().date()
            days_elapsed = max(1, (now - budget.start_date).days)
            days_remaining = max(0, (budget.end_date - now).days) if budget.end_date else 0
            
            # Calculate burn rate
            daily_burn_rate = current_spend / Decimal(str(days_elapsed)) if days_elapsed > 0 else Decimal('0')
            
            # Project spending
            if days_remaining > 0 and daily_burn_rate > 0:
                projected_spend = current_spend + (daily_burn_rate * Decimal(str(days_remaining)))
            else:
                projected_spend = current_spend
            
            # Calculate projected overage
            projected_overage = max(Decimal('0'), projected_spend - budget.amount)
            
            # Determine health status
            health_status = self._determine_health_status(utilization_percentage, projected_overage, budget.amount)
            
            return BudgetUtilization(
                budget_id=budget.id,
                budget_name=budget.name,
                budget_amount=budget.amount,
                current_spend=current_spend,
                utilization_percentage=utilization_percentage,
                remaining_amount=remaining_amount,
                days_elapsed=days_elapsed,
                days_remaining=days_remaining,
                daily_burn_rate=daily_burn_rate,
                projected_spend=projected_spend,
                projected_overage=projected_overage,
                health_status=health_status,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            self.logging_service.error(
                "Error calculating budget utilization",
                budget_id=str(budget.id),
                error=str(e)
            )
            raise
    
    async def generate_budget_forecast(self, budget: Budget) -> BudgetForecast:
        """Generate spending forecast for a budget"""
        try:
            # Get historical spending data
            historical_data = await self._get_historical_spending(budget, days=30)
            
            if len(historical_data) < 7:  # Need at least a week of data
                return self._create_simple_forecast(budget)
            
            # Use linear trend method for now (can be enhanced with more sophisticated methods)
            forecast = await self._forecast_linear_trend(budget, historical_data)
            
            return forecast
            
        except Exception as e:
            self.logging_service.error(
                "Error generating budget forecast",
                budget_id=str(budget.id),
                error=str(e)
            )
            return self._create_simple_forecast(budget)
    
    async def check_budget_alerts(self, 
                                budget: Budget, 
                                utilization: BudgetUtilization, 
                                forecast: BudgetForecast) -> List[BudgetAlert]:
        """Check if budget alerts should be triggered"""
        alerts = []
        
        try:
            # Check threshold-based alerts
            for threshold in budget.alert_thresholds:
                if utilization.utilization_percentage >= threshold:
                    # Check if alert already exists for this threshold
                    existing_alert = await self._get_existing_threshold_alert(budget.id, threshold)
                    
                    if not existing_alert:
                        alert = await self._create_threshold_alert(budget, utilization, threshold)
                        alerts.append(alert)
            
            # Check forecast-based alerts
            if forecast.probability_of_overage > 0.7:  # 70% probability of overage
                existing_forecast_alert = await self._get_existing_forecast_alert(budget.id)
                
                if not existing_forecast_alert:
                    alert = await self._create_forecast_alert(budget, utilization, forecast)
                    alerts.append(alert)
            
            # Check velocity-based alerts (rapid spending increase)
            if await self._detect_spending_velocity_anomaly(budget, utilization):
                existing_velocity_alert = await self._get_existing_velocity_alert(budget.id)
                
                if not existing_velocity_alert:
                    alert = await self._create_velocity_alert(budget, utilization)
                    alerts.append(alert)
            
            # Send notifications for new alerts
            for alert in alerts:
                await self._send_alert_notifications(alert, budget)
            
            return alerts
            
        except Exception as e:
            self.logging_service.error(
                "Error checking budget alerts",
                budget_id=str(budget.id),
                error=str(e)
            )
            return []
    
    async def _calculate_current_spend(self, budget: Budget) -> Decimal:
        """Calculate current spending for a budget based on its scope"""
        try:
            # Get cost data within budget period
            start_date = budget.start_date
            end_date = min(budget.end_date or date.today(), date.today())
            
            # For now, get all cost data (in practice, would filter by budget scope)
            # This is a simplified implementation - real implementation would use budget.scope_filters
            cost_data = await self.cost_data_repo.get_all(
                filters={
                    'cost_date__gte': start_date,
                    'cost_date__lte': end_date
                },
                limit=10000
            )
            
            # Sum up costs (simplified - would apply budget filters in practice)
            total_spend = sum(record.cost_amount for record in cost_data)
            
            return total_spend
            
        except Exception as e:
            self.logging_service.error(
                "Error calculating current spend",
                budget_id=str(budget.id),
                error=str(e)
            )
            return Decimal('0')
    
    def _determine_health_status(self, utilization_percentage: float, projected_overage: Decimal, budget_amount: Decimal) -> BudgetHealthStatus:
        """Determine budget health status based on utilization and projections"""
        if projected_overage > budget_amount * Decimal('0.2'):  # 20% overage
            return BudgetHealthStatus.CRITICAL
        elif utilization_percentage >= 100:
            return BudgetHealthStatus.OVER_BUDGET
        elif utilization_percentage >= self.at_risk_threshold * 100:
            return BudgetHealthStatus.AT_RISK
        else:
            return BudgetHealthStatus.HEALTHY
    
    async def _get_historical_spending(self, budget: Budget, days: int = 30) -> List[Tuple[date, Decimal]]:
        """Get historical daily spending data for a budget"""
        try:
            end_date = date.today()
            start_date = max(budget.start_date, end_date - timedelta(days=days))
            
            # Get cost data for the period
            cost_data = await self.cost_data_repo.get_all(
                filters={
                    'cost_date__gte': start_date,
                    'cost_date__lte': end_date
                },
                limit=10000
            )
            
            # Group by date and sum
            daily_spending = {}
            for record in cost_data:
                if record.cost_date not in daily_spending:
                    daily_spending[record.cost_date] = Decimal('0')
                daily_spending[record.cost_date] += record.cost_amount
            
            # Convert to sorted list
            return sorted(daily_spending.items())
            
        except Exception as e:
            self.logging_service.error(
                "Error getting historical spending",
                budget_id=str(budget.id),
                error=str(e)
            )
            return []
    
    def _create_simple_forecast(self, budget: Budget) -> BudgetForecast:
        """Create a simple forecast when insufficient data is available"""
        return BudgetForecast(
            budget_id=budget.id,
            forecast_date=datetime.utcnow(),
            method=ForecastMethod.LINEAR_TREND,
            confidence_level=0.3,  # Low confidence
            forecasted_spend_7_days=Decimal('0'),
            forecasted_spend_30_days=Decimal('0'),
            forecasted_spend_end_of_period=Decimal('0'),
            probability_of_overage=0.0,
            recommended_actions=["Insufficient data for accurate forecasting", "Monitor spending patterns"]
        )
    
    async def _forecast_linear_trend(self, budget: Budget, historical_data: List[Tuple[date, Decimal]]) -> BudgetForecast:
        """Generate forecast using linear trend analysis"""
        try:
            if len(historical_data) < 2:
                return self._create_simple_forecast(budget)
            
            # Calculate daily spending trend
            dates = [d for d, _ in historical_data]
            amounts = [float(amount) for _, amount in historical_data]
            
            # Simple linear regression
            n = len(amounts)
            x_values = list(range(n))
            
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(amounts)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, amounts))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean
            
            # Calculate confidence based on R-squared
            y_pred = [slope * x + intercept for x in x_values]
            ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(amounts))
            ss_tot = sum((y - y_mean) ** 2 for y in amounts)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0.1, min(0.9, r_squared))
            
            # Make predictions
            current_x = n
            forecast_7_days = Decimal(str(max(0, slope * (current_x + 7) + intercept)))
            forecast_30_days = Decimal(str(max(0, slope * (current_x + 30) + intercept)))
            
            # Forecast to end of budget period
            days_to_end = (budget.end_date - date.today()).days if budget.end_date else 30
            forecast_end_period = Decimal(str(max(0, slope * (current_x + days_to_end) + intercept)))
            
            # Calculate probability of overage
            current_spend = await self._calculate_current_spend(budget)
            total_projected = current_spend + forecast_end_period
            probability_overage = min(1.0, max(0.0, float(total_projected - budget.amount) / float(budget.amount)))
            
            # Generate recommendations
            recommendations = []
            if probability_overage > 0.7:
                recommendations.append("High probability of budget overage - implement cost controls")
            elif probability_overage > 0.3:
                recommendations.append("Monitor spending closely - potential budget risk")
            else:
                recommendations.append("Spending on track - continue monitoring")
            
            if slope > 0:
                recommendations.append("Spending trend is increasing - review cost drivers")
            
            return BudgetForecast(
                budget_id=budget.id,
                forecast_date=datetime.utcnow(),
                method=ForecastMethod.LINEAR_TREND,
                confidence_level=confidence,
                forecasted_spend_7_days=forecast_7_days,
                forecasted_spend_30_days=forecast_30_days,
                forecasted_spend_end_of_period=forecast_end_period,
                probability_of_overage=probability_overage,
                recommended_actions=recommendations
            )
            
        except Exception as e:
            self.logging_service.error(
                "Error in linear trend forecast",
                budget_id=str(budget.id),
                error=str(e)
            )
            return self._create_simple_forecast(budget)
    
    async def _get_existing_threshold_alert(self, budget_id: UUID, threshold: float) -> Optional[BudgetAlert]:
        """Check if threshold alert already exists"""
        try:
            alerts = await self.budget_alert_repo.get_alerts_by_budget(budget_id)
            
            for alert in alerts:
                if (alert.threshold_percentage == threshold and 
                    not alert.acknowledged and 
                    alert.alert_type == AlertType.THRESHOLD):
                    return alert
            
            return None
            
        except Exception as e:
            self.logging_service.error(
                "Error checking existing threshold alert",
                budget_id=str(budget_id),
                error=str(e)
            )
            return None
    
    async def _get_existing_forecast_alert(self, budget_id: UUID) -> Optional[BudgetAlert]:
        """Check if forecast alert already exists"""
        try:
            alerts = await self.budget_alert_repo.get_alerts_by_budget(budget_id)
            
            for alert in alerts:
                if (alert.alert_type == AlertType.FORECAST and 
                    not alert.acknowledged):
                    return alert
            
            return None
            
        except Exception as e:
            self.logging_service.error(
                "Error checking existing forecast alert",
                budget_id=str(budget_id),
                error=str(e)
            )
            return None
    
    async def _get_existing_velocity_alert(self, budget_id: UUID) -> Optional[BudgetAlert]:
        """Check if velocity alert already exists"""
        try:
            alerts = await self.budget_alert_repo.get_alerts_by_budget(budget_id)
            
            for alert in alerts:
                if (alert.alert_type.value == "velocity" and 
                    not alert.acknowledged):
                    return alert
            
            return None
            
        except Exception as e:
            self.logging_service.error(
                "Error checking existing velocity alert",
                budget_id=str(budget_id),
                error=str(e)
            )
            return None
    
    async def _create_threshold_alert(self, budget: Budget, utilization: BudgetUtilization, threshold: float) -> BudgetAlert:
        """Create a threshold-based budget alert"""
        alert = await self.budget_alert_repo.create(
            budget_id=budget.id,
            threshold_percentage=int(threshold),
            current_spend=utilization.current_spend,
            budget_amount=budget.amount,
            alert_type=AlertType.THRESHOLD,
            message=f"Budget '{budget.name}' has exceeded {threshold}% threshold ({utilization.utilization_percentage:.1f}% utilized)"
        )
        
        self.logging_service.info(
            "Created threshold alert",
            budget_id=str(budget.id),
            threshold=threshold,
            utilization=utilization.utilization_percentage
        )
        
        return alert
    
    async def _create_forecast_alert(self, budget: Budget, utilization: BudgetUtilization, forecast: BudgetForecast) -> BudgetAlert:
        """Create a forecast-based budget alert"""
        alert = await self.budget_alert_repo.create(
            budget_id=budget.id,
            threshold_percentage=None,
            current_spend=utilization.current_spend,
            budget_amount=budget.amount,
            alert_type=AlertType.FORECAST,
            message=f"Budget '{budget.name}' is projected to exceed budget with {forecast.probability_of_overage*100:.1f}% probability"
        )
        
        self.logging_service.info(
            "Created forecast alert",
            budget_id=str(budget.id),
            probability_overage=forecast.probability_of_overage
        )
        
        return alert
    
    async def _create_velocity_alert(self, budget: Budget, utilization: BudgetUtilization) -> BudgetAlert:
        """Create a velocity-based budget alert"""
        alert = await self.budget_alert_repo.create(
            budget_id=budget.id,
            threshold_percentage=None,
            current_spend=utilization.current_spend,
            budget_amount=budget.amount,
            alert_type=AlertType.ANOMALY,  # Using ANOMALY type for velocity alerts
            message=f"Budget '{budget.name}' has unusual spending velocity pattern detected"
        )
        
        self.logging_service.info(
            "Created velocity alert",
            budget_id=str(budget.id),
            daily_burn_rate=float(utilization.daily_burn_rate)
        )
        
        return alert
    
    async def _detect_spending_velocity_anomaly(self, budget: Budget, utilization: BudgetUtilization) -> bool:
        """Detect if spending velocity is anomalous"""
        try:
            # Get historical burn rates
            historical_data = await self._get_historical_spending(budget, days=14)
            
            if len(historical_data) < 7:
                return False
            
            # Calculate historical daily averages
            daily_amounts = [float(amount) for _, amount in historical_data]
            historical_avg = statistics.mean(daily_amounts)
            
            # Check if current burn rate is significantly higher
            current_burn_rate = float(utilization.daily_burn_rate)
            
            if historical_avg > 0:
                velocity_ratio = current_burn_rate / historical_avg
                return velocity_ratio > 2.0  # 200% increase in spending velocity
            
            return False
            
        except Exception as e:
            self.logging_service.error(
                "Error detecting velocity anomaly",
                budget_id=str(budget.id),
                error=str(e)
            )
            return False
    
    async def _send_alert_notifications(self, alert: BudgetAlert, budget: Budget) -> None:
        """Send notifications for a budget alert"""
        try:
            # Use the alert manager to send notifications
            await self.alert_manager.send_budget_alerts(budget, [alert])
            
        except Exception as e:
            self.logging_service.error(
                "Error sending alert notifications",
                alert_id=str(alert.id),
                budget_id=str(budget.id),
                error=str(e)
            )
    
    async def _cache_budget_utilization(self, budget_id: UUID, utilization: BudgetUtilization) -> None:
        """Cache budget utilization data"""
        try:
            cache_key = f"budget_utilization:{budget_id}:{date.today().isoformat()}"
            
            cache_data = {
                "budget_id": str(budget_id),
                "budget_name": utilization.budget_name,
                "utilization_percentage": utilization.utilization_percentage,
                "current_spend": float(utilization.current_spend),
                "budget_amount": float(utilization.budget_amount),
                "health_status": utilization.health_status.value,
                "daily_burn_rate": float(utilization.daily_burn_rate),
                "projected_overage": float(utilization.projected_overage),
                "last_updated": utilization.last_updated.isoformat()
            }
            
            await self.cache_service.set(cache_key, cache_data, ttl=3600)  # 1 hour
            
        except Exception as e:
            self.logging_service.warning(
                "Failed to cache budget utilization",
                budget_id=str(budget_id),
                error=str(e)
            )
    
    async def get_budget_utilization_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get budget utilization summary for the last N days"""
        try:
            summary_data = []
            
            for i in range(days):
                target_date = date.today() - timedelta(days=i)
                
                # Get cached utilization data for this date
                active_budgets = await self.budget_repo.get_active_budgets()
                
                day_summary = {
                    "date": target_date.isoformat(),
                    "total_budgets": len(active_budgets),
                    "budgets_at_risk": 0,
                    "budgets_over_budget": 0,
                    "total_utilization": 0.0,
                    "total_spend": 0.0,
                    "total_budget_amount": 0.0
                }
                
                for budget in active_budgets:
                    cache_key = f"budget_utilization:{budget.id}:{target_date.isoformat()}"
                    cached_data = await self.cache_service.get(cache_key)
                    
                    if cached_data:
                        if cached_data.get("health_status") == BudgetHealthStatus.AT_RISK.value:
                            day_summary["budgets_at_risk"] += 1
                        elif cached_data.get("health_status") in [BudgetHealthStatus.OVER_BUDGET.value, BudgetHealthStatus.CRITICAL.value]:
                            day_summary["budgets_over_budget"] += 1
                        
                        day_summary["total_spend"] += cached_data.get("current_spend", 0)
                        day_summary["total_budget_amount"] += cached_data.get("budget_amount", 0)
                
                if day_summary["total_budget_amount"] > 0:
                    day_summary["total_utilization"] = (day_summary["total_spend"] / day_summary["total_budget_amount"]) * 100
                
                summary_data.append(day_summary)
            
            return {
                "summary_period_days": days,
                "daily_summaries": summary_data,
                "trends": self._calculate_utilization_trends(summary_data)
            }
            
        except Exception as e:
            self.logging_service.error("Error getting budget utilization summary", error=str(e))
            return {"error": str(e)}
    
    def _calculate_utilization_trends(self, daily_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate utilization trends from daily summaries"""
        if len(daily_summaries) < 2:
            return {}
        
        # Calculate trends
        utilizations = [day["total_utilization"] for day in daily_summaries if day["total_utilization"] > 0]
        
        if len(utilizations) < 2:
            return {}
        
        # Simple trend calculation
        trend_direction = "increasing" if utilizations[-1] > utilizations[0] else "decreasing"
        trend_magnitude = abs(utilizations[-1] - utilizations[0])
        
        return {
            "trend_direction": trend_direction,
            "trend_magnitude": trend_magnitude,
            "average_utilization": statistics.mean(utilizations),
            "max_utilization": max(utilizations),
            "min_utilization": min(utilizations)
        }