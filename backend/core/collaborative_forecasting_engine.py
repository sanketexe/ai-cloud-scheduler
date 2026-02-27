"""
Collaborative Forecasting Engine for Real-Time FinOps Workspace

This module implements multi-user forecast input collection, confidence interval
calculation based on collective estimates, and scenario management with version
history and comparison capabilities.
"""

import uuid
import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import numpy as np
from scipy import stats

from .database import get_db_session
from .collaborative_budget_planner import CollaborativeBudget, collaborative_budget_planner
from .state_synchronization import sync_engine, SyncEvent, SyncEventType
from .operational_transformation import Operation, OperationType
from .redis_config import redis_manager

logger = logging.getLogger(__name__)

class ForecastMethod(Enum):
    """Forecasting methods"""
    CONSENSUS = "consensus"  # Average of all inputs
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by confidence/expertise
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    DELPHI = "delphi"  # Delphi method with rounds
    HISTORICAL_TREND = "historical_trend"  # Based on historical data
    MACHINE_LEARNING = "machine_learning"  # ML-based prediction

class ForecastPeriod(Enum):
    """Forecast time periods"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"

class ScenarioType(Enum):
    """Budget scenario types"""
    OPTIMISTIC = "optimistic"
    REALISTIC = "realistic"
    PESSIMISTIC = "pessimistic"
    CUSTOM = "custom"

@dataclass
class ForecastInput:
    """Individual forecast input from a participant"""
    input_id: str
    session_id: str
    user_id: str
    forecast_target: str  # What is being forecasted (category, line_item, total)
    target_path: str  # Path to the budget element
    period: ForecastPeriod
    period_start: datetime
    period_end: datetime
    estimated_value: Decimal
    confidence_level: float  # 0.0 to 1.0
    reasoning: str
    assumptions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    methodology: str = ""
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CollectiveForecast:
    """Collective forecast result from multiple inputs"""
    forecast_id: str
    session_id: str
    forecast_target: str
    target_path: str
    period: ForecastPeriod
    period_start: datetime
    period_end: datetime
    method: ForecastMethod
    
    # Statistical results
    consensus_value: Decimal
    confidence_interval_lower: Decimal
    confidence_interval_upper: Decimal
    confidence_level: float  # Overall confidence (0.0 to 1.0)
    standard_deviation: Decimal
    
    # Input analysis
    input_count: int
    min_estimate: Decimal
    max_estimate: Decimal
    median_estimate: Decimal
    
    # Participant data
    participant_inputs: List[ForecastInput] = field(default_factory=list)
    expert_weights: Dict[str, float] = field(default_factory=dict)  # user_id -> weight
    
    # Metadata
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BudgetScenario:
    """Budget scenario with forecasts and comparisons"""
    scenario_id: str
    session_id: str
    name: str
    description: str
    scenario_type: ScenarioType
    base_budget_id: str
    
    # Scenario data
    forecasts: List[CollectiveForecast] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Calculated totals
    total_forecast: Decimal = Decimal('0')
    variance_from_base: Decimal = Decimal('0')
    variance_percentage: float = 0.0
    
    # Version control
    version: int = 1
    parent_scenario_id: Optional[str] = None
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified_by: str = ""
    last_modified_at: datetime = field(default_factory=datetime.utcnow)
    
    # Collaboration metadata
    contributors: List[str] = field(default_factory=list)
    approval_status: str = "draft"  # draft, under_review, approved, rejected
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

@dataclass
class ScenarioComparison:
    """Comparison between multiple budget scenarios"""
    comparison_id: str
    session_id: str
    name: str
    scenarios: List[str]  # scenario_ids
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)
    variance_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

class CollaborativeForecastingEngine:
    """
    Multi-user forecast input collection with confidence interval calculation
    based on collective estimates and scenario management
    """
    
    def __init__(self):
        self.active_forecasts: Dict[str, Dict[str, CollectiveForecast]] = {}  # session_id -> {target_path -> forecast}
        self.forecast_inputs: Dict[str, List[ForecastInput]] = {}  # session_id -> inputs
        self.scenarios: Dict[str, List[BudgetScenario]] = {}  # session_id -> scenarios
        self.forecast_locks: Dict[str, asyncio.Lock] = {}  # session_id -> lock
        
    async def collect_forecast_input(self, session_id: str, forecast_input: ForecastInput) -> Dict[str, Any]:
        """
        Collect forecast input from a participant and update collective forecast
        
        Args:
            session_id: ID of the collaborative session
            forecast_input: Forecast input from participant
            
        Returns:
            Dict containing updated collective forecast
        """
        try:
            if session_id not in self.forecast_locks:
                self.forecast_locks[session_id] = asyncio.Lock()
            
            async with self.forecast_locks[session_id]:
                # Initialize session data if needed
                if session_id not in self.forecast_inputs:
                    self.forecast_inputs[session_id] = []
                if session_id not in self.active_forecasts:
                    self.active_forecasts[session_id] = {}
                
                # Add or update forecast input
                existing_input = None
                for i, existing in enumerate(self.forecast_inputs[session_id]):
                    if (existing.user_id == forecast_input.user_id and 
                        existing.target_path == forecast_input.target_path and
                        existing.period == forecast_input.period):
                        existing_input = i
                        break
                
                if existing_input is not None:
                    # Update existing input
                    self.forecast_inputs[session_id][existing_input] = forecast_input
                    logger.info(f"Updated forecast input from user {forecast_input.user_id}")
                else:
                    # Add new input
                    self.forecast_inputs[session_id].append(forecast_input)
                    logger.info(f"Added new forecast input from user {forecast_input.user_id}")
                
                # Recalculate collective forecast
                collective_forecast = await self._calculate_collective_forecast(
                    session_id, forecast_input.target_path, forecast_input.period
                )
                
                # Store updated collective forecast
                forecast_key = f"{forecast_input.target_path}_{forecast_input.period.value}"
                self.active_forecasts[session_id][forecast_key] = collective_forecast
                
                # Cache forecast data
                await self._cache_forecast_data(session_id)
                
                # Broadcast forecast update
                await self._broadcast_forecast_update(session_id, collective_forecast, forecast_input.user_id)
                
                return {
                    "success": True,
                    "collective_forecast": self._serialize_collective_forecast(collective_forecast),
                    "input_count": collective_forecast.input_count,
                    "confidence_interval": {
                        "lower": str(collective_forecast.confidence_interval_lower),
                        "upper": str(collective_forecast.confidence_interval_upper),
                        "confidence_level": collective_forecast.confidence_level
                    }
                }
                
        except Exception as e:
            logger.error(f"Error collecting forecast input: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_scenario(self, session_id: str, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new budget scenario with version history
        
        Args:
            session_id: ID of the collaborative session
            scenario_config: Scenario configuration
            
        Returns:
            Dict containing created scenario
        """
        try:
            if session_id not in self.scenarios:
                self.scenarios[session_id] = []
            
            # Get base budget for variance calculation
            budget_state = await collaborative_budget_planner.get_budget_state(session_id)
            if not budget_state:
                return {"success": False, "error": "No active budget found"}
            
            base_budget = budget_state["budget"]
            
            # Create scenario
            scenario = BudgetScenario(
                scenario_id=str(uuid.uuid4()),
                session_id=session_id,
                name=scenario_config.get("name", "New Scenario"),
                description=scenario_config.get("description", ""),
                scenario_type=ScenarioType(scenario_config.get("type", "realistic")),
                base_budget_id=base_budget["budget_id"],
                assumptions=scenario_config.get("assumptions", []),
                created_by=scenario_config.get("created_by", ""),
                contributors=[scenario_config.get("created_by", "")]
            )
            
            # Copy relevant forecasts to scenario
            if session_id in self.active_forecasts:
                for forecast in self.active_forecasts[session_id].values():
                    scenario.forecasts.append(forecast)
            
            # Calculate scenario totals
            await self._calculate_scenario_totals(scenario, base_budget)
            
            # Add to scenarios list
            self.scenarios[session_id].append(scenario)
            
            # Cache scenario data
            await self._cache_scenario_data(session_id)
            
            # Broadcast scenario creation
            await self._broadcast_scenario_update(session_id, scenario, scenario.created_by, "created")
            
            logger.info(f"Created scenario {scenario.name} for session {session_id}")
            
            return {
                "success": True,
                "scenario": self._serialize_scenario(scenario)
            }
            
        except Exception as e:
            logger.error(f"Error creating scenario: {e}")
            return {"success": False, "error": str(e)}
    
    async def compare_scenarios(self, session_id: str, scenario_ids: List[str], 
                              comparison_name: str, created_by: str) -> Dict[str, Any]:
        """
        Compare multiple budget scenarios
        
        Args:
            session_id: ID of the collaborative session
            scenario_ids: List of scenario IDs to compare
            comparison_name: Name for the comparison
            created_by: User creating the comparison
            
        Returns:
            Dict containing scenario comparison results
        """
        try:
            if session_id not in self.scenarios:
                return {"success": False, "error": "No scenarios found"}
            
            # Get scenarios to compare
            scenarios_to_compare = []
            for scenario in self.scenarios[session_id]:
                if scenario.scenario_id in scenario_ids:
                    scenarios_to_compare.append(scenario)
            
            if len(scenarios_to_compare) < 2:
                return {"success": False, "error": "At least 2 scenarios required for comparison"}
            
            # Perform comparison analysis
            comparison = ScenarioComparison(
                comparison_id=str(uuid.uuid4()),
                session_id=session_id,
                name=comparison_name,
                scenarios=scenario_ids,
                created_by=created_by
            )
            
            # Calculate comparison metrics
            comparison.comparison_metrics = await self._calculate_comparison_metrics(scenarios_to_compare)
            comparison.variance_analysis = await self._analyze_scenario_variance(scenarios_to_compare)
            comparison.risk_analysis = await self._analyze_scenario_risks(scenarios_to_compare)
            comparison.recommendations = await self._generate_scenario_recommendations(scenarios_to_compare)
            
            # Broadcast comparison results
            await self._broadcast_comparison_results(session_id, comparison, created_by)
            
            return {
                "success": True,
                "comparison": self._serialize_comparison(comparison)
            }
            
        except Exception as e:
            logger.error(f"Error comparing scenarios: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_forecast_state(self, session_id: str) -> Dict[str, Any]:
        """Get current forecasting state for a session"""
        try:
            return {
                "active_forecasts": {
                    key: self._serialize_collective_forecast(forecast)
                    for key, forecast in self.active_forecasts.get(session_id, {}).items()
                },
                "forecast_inputs": [
                    self._serialize_forecast_input(input_data)
                    for input_data in self.forecast_inputs.get(session_id, [])
                ],
                "scenarios": [
                    self._serialize_scenario(scenario)
                    for scenario in self.scenarios.get(session_id, [])
                ],
                "input_statistics": await self._calculate_input_statistics(session_id)
            }
            
        except Exception as e:
            logger.error(f"Error getting forecast state: {e}")
            return {}
    
    # Private helper methods
    
    async def _calculate_collective_forecast(self, session_id: str, target_path: str, 
                                           period: ForecastPeriod) -> CollectiveForecast:
        """Calculate collective forecast from individual inputs"""
        try:
            # Get relevant inputs
            relevant_inputs = [
                input_data for input_data in self.forecast_inputs.get(session_id, [])
                if input_data.target_path == target_path and input_data.period == period
            ]
            
            if not relevant_inputs:
                # Create empty forecast
                return CollectiveForecast(
                    forecast_id=str(uuid.uuid4()),
                    session_id=session_id,
                    forecast_target=target_path.split('.')[-1],
                    target_path=target_path,
                    period=period,
                    period_start=datetime.utcnow(),
                    period_end=datetime.utcnow() + timedelta(days=30),
                    method=ForecastMethod.CONSENSUS,
                    consensus_value=Decimal('0'),
                    confidence_interval_lower=Decimal('0'),
                    confidence_interval_upper=Decimal('0'),
                    confidence_level=0.0,
                    standard_deviation=Decimal('0'),
                    input_count=0,
                    min_estimate=Decimal('0'),
                    max_estimate=Decimal('0'),
                    median_estimate=Decimal('0')
                )
            
            # Extract values and weights
            values = [float(input_data.estimated_value) for input_data in relevant_inputs]
            weights = [input_data.confidence_level for input_data in relevant_inputs]
            
            # Calculate statistical measures
            consensus_value = self._calculate_weighted_average(values, weights)
            confidence_interval = self._calculate_confidence_interval(values, weights)
            overall_confidence = self._calculate_overall_confidence(relevant_inputs)
            
            # Create collective forecast
            forecast = CollectiveForecast(
                forecast_id=str(uuid.uuid4()),
                session_id=session_id,
                forecast_target=target_path.split('.')[-1],
                target_path=target_path,
                period=period,
                period_start=relevant_inputs[0].period_start,
                period_end=relevant_inputs[0].period_end,
                method=ForecastMethod.WEIGHTED_AVERAGE,
                consensus_value=Decimal(str(consensus_value)),
                confidence_interval_lower=Decimal(str(confidence_interval[0])),
                confidence_interval_upper=Decimal(str(confidence_interval[1])),
                confidence_level=overall_confidence,
                standard_deviation=Decimal(str(statistics.stdev(values) if len(values) > 1 else 0)),
                input_count=len(relevant_inputs),
                min_estimate=Decimal(str(min(values))),
                max_estimate=Decimal(str(max(values))),
                median_estimate=Decimal(str(statistics.median(values))),
                participant_inputs=relevant_inputs.copy()
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error calculating collective forecast: {e}")
            raise
    
    def _calculate_weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted average of forecast values"""
        if not values or not weights:
            return 0.0
        
        total_weight = sum(weights)
        if total_weight == 0:
            return statistics.mean(values)
        
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / total_weight
    
    def _calculate_confidence_interval(self, values: List[float], weights: List[float], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for forecast values"""
        if len(values) < 2:
            return (values[0] if values else 0.0, values[0] if values else 0.0)
        
        try:
            # Use weighted statistics if possible
            mean = self._calculate_weighted_average(values, weights)
            
            # Calculate weighted standard deviation
            total_weight = sum(weights)
            if total_weight > 0:
                variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_weight
                std_dev = variance ** 0.5
            else:
                std_dev = statistics.stdev(values)
            
            # Calculate confidence interval using t-distribution
            n = len(values)
            t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin_of_error = t_value * std_dev / (n ** 0.5)
            
            return (mean - margin_of_error, mean + margin_of_error)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            # Fallback to simple range
            return (min(values), max(values))
    
    def _calculate_overall_confidence(self, inputs: List[ForecastInput]) -> float:
        """Calculate overall confidence from individual confidence levels"""
        if not inputs:
            return 0.0
        
        # Use harmonic mean for conservative confidence estimate
        confidence_levels = [input_data.confidence_level for input_data in inputs]
        
        # Filter out zero confidence levels
        non_zero_confidence = [c for c in confidence_levels if c > 0]
        if not non_zero_confidence:
            return 0.0
        
        # Calculate harmonic mean
        harmonic_mean = len(non_zero_confidence) / sum(1/c for c in non_zero_confidence)
        
        # Apply group size factor (more participants = higher confidence)
        group_factor = min(1.0, len(inputs) / 5.0)  # Max benefit at 5+ participants
        
        return min(1.0, harmonic_mean * (0.7 + 0.3 * group_factor))
    
    async def _calculate_scenario_totals(self, scenario: BudgetScenario, base_budget: Dict[str, Any]):
        """Calculate scenario totals and variance from base budget"""
        try:
            # Calculate total forecast from all forecasts in scenario
            scenario.total_forecast = sum(
                forecast.consensus_value for forecast in scenario.forecasts
            )
            
            # Calculate variance from base budget
            base_total = Decimal(str(base_budget.get("total_amount", 0)))
            scenario.variance_from_base = scenario.total_forecast - base_total
            
            if base_total > 0:
                scenario.variance_percentage = float(scenario.variance_from_base / base_total * 100)
            else:
                scenario.variance_percentage = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating scenario totals: {e}")
    
    async def _calculate_comparison_metrics(self, scenarios: List[BudgetScenario]) -> Dict[str, Any]:
        """Calculate metrics for scenario comparison"""
        try:
            totals = [float(scenario.total_forecast) for scenario in scenarios]
            variances = [scenario.variance_percentage for scenario in scenarios]
            
            return {
                "total_range": {
                    "min": min(totals),
                    "max": max(totals),
                    "spread": max(totals) - min(totals)
                },
                "variance_range": {
                    "min": min(variances),
                    "max": max(variances),
                    "spread": max(variances) - min(variances)
                },
                "average_total": statistics.mean(totals),
                "median_total": statistics.median(totals),
                "standard_deviation": statistics.stdev(totals) if len(totals) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating comparison metrics: {e}")
            return {}
    
    async def _analyze_scenario_variance(self, scenarios: List[BudgetScenario]) -> Dict[str, Any]:
        """Analyze variance between scenarios"""
        try:
            variance_analysis = {
                "high_variance_scenarios": [],
                "low_variance_scenarios": [],
                "variance_drivers": []
            }
            
            # Identify high and low variance scenarios
            for scenario in scenarios:
                if abs(scenario.variance_percentage) > 20:
                    variance_analysis["high_variance_scenarios"].append({
                        "scenario_id": scenario.scenario_id,
                        "name": scenario.name,
                        "variance_percentage": scenario.variance_percentage
                    })
                elif abs(scenario.variance_percentage) < 5:
                    variance_analysis["low_variance_scenarios"].append({
                        "scenario_id": scenario.scenario_id,
                        "name": scenario.name,
                        "variance_percentage": scenario.variance_percentage
                    })
            
            return variance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing scenario variance: {e}")
            return {}
    
    async def _analyze_scenario_risks(self, scenarios: List[BudgetScenario]) -> Dict[str, Any]:
        """Analyze risks across scenarios"""
        try:
            risk_analysis = {
                "common_risks": [],
                "scenario_specific_risks": {},
                "risk_mitigation_suggestions": []
            }
            
            # Collect all risk factors
            all_risks = []
            for scenario in scenarios:
                for forecast in scenario.forecasts:
                    for input_data in forecast.participant_inputs:
                        all_risks.extend(input_data.risk_factors)
            
            # Find common risks
            risk_counts = {}
            for risk in all_risks:
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            # Risks mentioned in multiple scenarios are common
            common_threshold = len(scenarios) // 2 + 1
            risk_analysis["common_risks"] = [
                risk for risk, count in risk_counts.items() 
                if count >= common_threshold
            ]
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing scenario risks: {e}")
            return {}
    
    async def _generate_scenario_recommendations(self, scenarios: List[BudgetScenario]) -> List[str]:
        """Generate recommendations based on scenario analysis"""
        try:
            recommendations = []
            
            # Analyze scenario spread
            totals = [float(scenario.total_forecast) for scenario in scenarios]
            spread = max(totals) - min(totals)
            avg_total = statistics.mean(totals)
            
            if spread > avg_total * 0.3:  # High uncertainty
                recommendations.append(
                    "High variance between scenarios suggests significant uncertainty. "
                    "Consider gathering more data or refining assumptions."
                )
            
            # Check for optimistic bias
            optimistic_scenarios = [s for s in scenarios if s.scenario_type == ScenarioType.OPTIMISTIC]
            if optimistic_scenarios and len(optimistic_scenarios) > len(scenarios) / 2:
                recommendations.append(
                    "Consider adding more conservative scenarios to balance optimistic projections."
                )
            
            # Budget constraint recommendations
            over_budget_scenarios = [s for s in scenarios if s.variance_percentage > 10]
            if over_budget_scenarios:
                recommendations.append(
                    f"{len(over_budget_scenarios)} scenario(s) exceed budget by >10%. "
                    "Review assumptions and consider cost reduction measures."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _calculate_input_statistics(self, session_id: str) -> Dict[str, Any]:
        """Calculate statistics about forecast inputs"""
        try:
            inputs = self.forecast_inputs.get(session_id, [])
            
            if not inputs:
                return {}
            
            # Participant statistics
            participants = set(input_data.user_id for input_data in inputs)
            
            # Confidence statistics
            confidence_levels = [input_data.confidence_level for input_data in inputs]
            
            return {
                "total_inputs": len(inputs),
                "unique_participants": len(participants),
                "average_confidence": statistics.mean(confidence_levels),
                "confidence_range": {
                    "min": min(confidence_levels),
                    "max": max(confidence_levels)
                },
                "inputs_per_participant": len(inputs) / len(participants) if participants else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating input statistics: {e}")
            return {}
    
    async def _broadcast_forecast_update(self, session_id: str, forecast: CollectiveForecast, user_id: str):
        """Broadcast forecast update to all participants"""
        try:
            update_event = SyncEvent(
                event_id=f"forecast_update_{forecast.forecast_id}",
                event_type=SyncEventType.STATE_UPDATE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "operation_type": "forecast_update",
                    "forecast": self._serialize_collective_forecast(forecast)
                },
                priority=2
            )
            
            await sync_engine._queue_sync_event(session_id, update_event)
            
        except Exception as e:
            logger.error(f"Error broadcasting forecast update: {e}")
    
    async def _broadcast_scenario_update(self, session_id: str, scenario: BudgetScenario, 
                                       user_id: str, action: str):
        """Broadcast scenario update to all participants"""
        try:
            update_event = SyncEvent(
                event_id=f"scenario_{action}_{scenario.scenario_id}",
                event_type=SyncEventType.STATE_UPDATE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "operation_type": f"scenario_{action}",
                    "scenario": self._serialize_scenario(scenario)
                },
                priority=2
            )
            
            await sync_engine._queue_sync_event(session_id, update_event)
            
        except Exception as e:
            logger.error(f"Error broadcasting scenario update: {e}")
    
    async def _broadcast_comparison_results(self, session_id: str, comparison: ScenarioComparison, user_id: str):
        """Broadcast scenario comparison results"""
        try:
            update_event = SyncEvent(
                event_id=f"comparison_{comparison.comparison_id}",
                event_type=SyncEventType.STATE_UPDATE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "operation_type": "scenario_comparison",
                    "comparison": self._serialize_comparison(comparison)
                },
                priority=2
            )
            
            await sync_engine._queue_sync_event(session_id, update_event)
            
        except Exception as e:
            logger.error(f"Error broadcasting comparison results: {e}")
    
    async def _cache_forecast_data(self, session_id: str):
        """Cache forecast data in Redis"""
        try:
            cache_key = f"collaborative_forecasts:{session_id}"
            forecast_data = {
                "active_forecasts": {
                    key: self._serialize_collective_forecast(forecast)
                    for key, forecast in self.active_forecasts.get(session_id, {}).items()
                },
                "forecast_inputs": [
                    self._serialize_forecast_input(input_data)
                    for input_data in self.forecast_inputs.get(session_id, [])
                ]
            }
            
            client = await redis_manager.get_async_client()
            import json
            await client.set(cache_key, json.dumps(forecast_data), ex=3600)
        except Exception as e:
            logger.error(f"Error caching forecast data: {e}")
    
    async def _cache_scenario_data(self, session_id: str):
        """Cache scenario data in Redis"""
        try:
            cache_key = f"collaborative_scenarios:{session_id}"
            scenario_data = [
                self._serialize_scenario(scenario)
                for scenario in self.scenarios.get(session_id, [])
            ]
            
            client = await redis_manager.get_async_client()
            import json
            await client.set(cache_key, json.dumps(scenario_data), ex=3600)
        except Exception as e:
            logger.error(f"Error caching scenario data: {e}")
    
    # Serialization methods
    
    def _serialize_forecast_input(self, input_data: ForecastInput) -> Dict[str, Any]:
        """Serialize forecast input to dictionary"""
        return {
            "input_id": input_data.input_id,
            "session_id": input_data.session_id,
            "user_id": input_data.user_id,
            "forecast_target": input_data.forecast_target,
            "target_path": input_data.target_path,
            "period": input_data.period.value,
            "period_start": input_data.period_start.isoformat(),
            "period_end": input_data.period_end.isoformat(),
            "estimated_value": str(input_data.estimated_value),
            "confidence_level": input_data.confidence_level,
            "reasoning": input_data.reasoning,
            "assumptions": input_data.assumptions,
            "risk_factors": input_data.risk_factors,
            "methodology": input_data.methodology,
            "supporting_data": input_data.supporting_data,
            "created_at": input_data.created_at.isoformat(),
            "updated_at": input_data.updated_at.isoformat()
        }
    
    def _serialize_collective_forecast(self, forecast: CollectiveForecast) -> Dict[str, Any]:
        """Serialize collective forecast to dictionary"""
        return {
            "forecast_id": forecast.forecast_id,
            "session_id": forecast.session_id,
            "forecast_target": forecast.forecast_target,
            "target_path": forecast.target_path,
            "period": forecast.period.value,
            "period_start": forecast.period_start.isoformat(),
            "period_end": forecast.period_end.isoformat(),
            "method": forecast.method.value,
            "consensus_value": str(forecast.consensus_value),
            "confidence_interval_lower": str(forecast.confidence_interval_lower),
            "confidence_interval_upper": str(forecast.confidence_interval_upper),
            "confidence_level": forecast.confidence_level,
            "standard_deviation": str(forecast.standard_deviation),
            "input_count": forecast.input_count,
            "min_estimate": str(forecast.min_estimate),
            "max_estimate": str(forecast.max_estimate),
            "median_estimate": str(forecast.median_estimate),
            "participant_inputs": [
                self._serialize_forecast_input(input_data) 
                for input_data in forecast.participant_inputs
            ],
            "expert_weights": forecast.expert_weights,
            "calculation_metadata": forecast.calculation_metadata,
            "created_at": forecast.created_at.isoformat(),
            "updated_at": forecast.updated_at.isoformat()
        }
    
    def _serialize_scenario(self, scenario: BudgetScenario) -> Dict[str, Any]:
        """Serialize budget scenario to dictionary"""
        return {
            "scenario_id": scenario.scenario_id,
            "session_id": scenario.session_id,
            "name": scenario.name,
            "description": scenario.description,
            "scenario_type": scenario.scenario_type.value,
            "base_budget_id": scenario.base_budget_id,
            "forecasts": [
                self._serialize_collective_forecast(forecast) 
                for forecast in scenario.forecasts
            ],
            "assumptions": scenario.assumptions,
            "risk_assessment": scenario.risk_assessment,
            "total_forecast": str(scenario.total_forecast),
            "variance_from_base": str(scenario.variance_from_base),
            "variance_percentage": scenario.variance_percentage,
            "version": scenario.version,
            "parent_scenario_id": scenario.parent_scenario_id,
            "created_by": scenario.created_by,
            "created_at": scenario.created_at.isoformat(),
            "last_modified_by": scenario.last_modified_by,
            "last_modified_at": scenario.last_modified_at.isoformat(),
            "contributors": scenario.contributors,
            "approval_status": scenario.approval_status,
            "approved_by": scenario.approved_by,
            "approved_at": scenario.approved_at.isoformat() if scenario.approved_at else None
        }
    
    def _serialize_comparison(self, comparison: ScenarioComparison) -> Dict[str, Any]:
        """Serialize scenario comparison to dictionary"""
        return {
            "comparison_id": comparison.comparison_id,
            "session_id": comparison.session_id,
            "name": comparison.name,
            "scenarios": comparison.scenarios,
            "comparison_metrics": comparison.comparison_metrics,
            "variance_analysis": comparison.variance_analysis,
            "risk_analysis": comparison.risk_analysis,
            "recommendations": comparison.recommendations,
            "created_by": comparison.created_by,
            "created_at": comparison.created_at.isoformat()
        }

# Global collaborative forecasting engine instance
collaborative_forecasting_engine = CollaborativeForecastingEngine()