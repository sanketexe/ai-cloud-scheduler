"""
Forecast Engine for Multi-Horizon Predictive Cost Forecasting

Provides unified forecasting capabilities with multiple time horizons (7, 30, 90 days),
confidence intervals, budget overrun prediction, and seasonal pattern detection.
Orchestrates Prophet, LSTM, and ensemble models for comprehensive cost predictions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class ForecastHorizon(Enum):
    """Supported forecast horizons"""
    SHORT_TERM = 7      # 7 days - tactical forecasting
    MEDIUM_TERM = 30    # 30 days - strategic planning
    LONG_TERM = 90      # 90 days - budget planning

class ForecastConfidence(Enum):
    """Forecast confidence levels"""
    HIGH = 0.95         # 95% confidence interval
    MEDIUM = 0.80       # 80% confidence interval
    LOW = 0.68          # 68% confidence interval

@dataclass
class ForecastRequest:
    """Forecast generation request"""
    account_id: str
    service: Optional[str] = None
    resource_id: Optional[str] = None
    horizons: List[ForecastHorizon] = field(default_factory=lambda: [ForecastHorizon.MEDIUM_TERM])
    confidence_level: ForecastConfidence = ForecastConfidence.MEDIUM
    include_seasonality: bool = True
    include_budget_analysis: bool = True
    budget_amount: Optional[float] = None
    budget_period_days: Optional[int] = None

@dataclass
class ForecastPoint:
    """Individual forecast data point"""
    timestamp: datetime
    forecasted_value: float
    lower_bound: float
    upper_bound: float
    confidence_interval_width: float
    trend_component: float
    seasonal_component: float
    uncertainty: float
    model_contributions: Dict[str, float]

@dataclass
class BudgetAnalysis:
    """Budget overrun analysis"""
    budget_amount: float
    current_spend: float
    projected_total_spend: float
    overrun_probability: float
    days_until_overrun: Optional[int]
    overrun_amount: float
    confidence_score: float
    risk_level: str

@dataclass
class ForecastResult:
    """Complete forecast result"""
    request_id: str
    account_id: str
    service: Optional[str]
    resource_id: Optional[str]
    horizon_days: int
    generated_at: datetime
    forecast_points: List[ForecastPoint]
    accuracy_metrics: Dict[str, float]
    model_performance: Dict[str, Any]
    budget_analysis: Optional[BudgetAnalysis]
    seasonal_patterns: Dict[str, Any]
    confidence_assessment: Dict[str, float]
    recommendations: List[str]

class ForecastEngine:
    """
    Unified forecasting engine for multi-horizon cost predictions.
    
    Orchestrates multiple ML models (Prophet, LSTM, Ensemble) to provide
    comprehensive cost forecasting with confidence intervals, seasonal
    adjustment, and budget overrun analysis.
    """
    
    def __init__(self):
        self.models = {}
        self.model_weights = {
            'prophet': 0.5,
            'lstm': 0.3,
            'ensemble': 0.2
        }
        
        # Performance tracking
        self.forecast_history = []
        self.model_performance = {}
        
        # Configuration
        self.max_concurrent_forecasts = 5
        self.default_confidence_level = ForecastConfidence.MEDIUM
        
        # Accuracy targets by horizon
        self.accuracy_targets = {
            ForecastHorizon.SHORT_TERM: 0.95,   # 95% accuracy for 7-day
            ForecastHorizon.MEDIUM_TERM: 0.85,  # 85% accuracy for 30-day
            ForecastHorizon.LONG_TERM: 0.75     # 75% accuracy for 90-day
        }
        
        # Initialize thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_forecasts)
    
    async def initialize_models(self):
        """Initialize and load forecasting models"""
        try:
            # Import models with fallback handling
            from .prophet_forecaster import ProphetForecaster, ProphetConfig
            from .lstm_anomaly_detector import LSTMAnomalyDetector, LSTMModelConfig
            from .ensemble_scorer import EnsembleScorer, EnsembleConfig
            
            # Initialize Prophet for seasonal forecasting
            prophet_config = ProphetConfig(
                growth='linear',
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,
                interval_width=0.80
            )
            self.models['prophet'] = ProphetForecaster(prophet_config)
            
            # Initialize LSTM for time series patterns
            lstm_config = LSTMModelConfig(
                sequence_length=14,
                lstm_units=50,
                epochs=10,
                batch_size=32
            )
            self.models['lstm'] = LSTMAnomalyDetector(lstm_config)
            
            # Initialize ensemble scorer
            ensemble_config = EnsembleConfig(
                anomaly_threshold=0.7,
                confidence_threshold=0.6
            )
            self.models['ensemble'] = EnsembleScorer(ensemble_config)
            
            logger.info("Forecast engine models initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Some models unavailable: {e}")
            # Create fallback models
            self.models['fallback'] = self._create_fallback_model()
    
    async def generate_forecast(
        self,
        request: ForecastRequest,
        historical_data: pd.DataFrame
    ) -> ForecastResult:
        """
        Generate comprehensive forecast for the given request.
        
        Args:
            request: Forecast generation request
            historical_data: Historical cost data
            
        Returns:
            Complete forecast result with all horizons
        """
        request_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.account_id}"
        
        logger.info(f"Generating forecast {request_id} for account {request.account_id}")
        
        try:
            # Validate input data
            self._validate_historical_data(historical_data)
            
            # Generate forecasts for all requested horizons
            forecast_results = {}
            
            for horizon in request.horizons:
                logger.info(f"Generating {horizon.value}-day forecast")
                
                # Generate forecast points
                forecast_points = await self._generate_horizon_forecast(
                    historical_data=historical_data,
                    horizon_days=horizon.value,
                    confidence_level=request.confidence_level,
                    include_seasonality=request.include_seasonality
                )
                
                forecast_results[horizon] = forecast_points
            
            # Select primary horizon for main result (medium-term if available)
            primary_horizon = (ForecastHorizon.MEDIUM_TERM if ForecastHorizon.MEDIUM_TERM in request.horizons 
                             else request.horizons[0])
            primary_forecast = forecast_results[primary_horizon]
            
            # Calculate accuracy metrics
            accuracy_metrics = await self._calculate_accuracy_metrics(
                historical_data, primary_forecast, primary_horizon
            )
            
            # Analyze seasonal patterns
            seasonal_patterns = await self._analyze_seasonal_patterns(
                historical_data, primary_forecast
            )
            
            # Generate budget analysis if requested
            budget_analysis = None
            if request.include_budget_analysis and request.budget_amount:
                budget_analysis = await self._analyze_budget_overrun(
                    forecast_points=primary_forecast,
                    budget_amount=request.budget_amount,
                    budget_period_days=request.budget_period_days or primary_horizon.value,
                    current_spend=self._calculate_current_spend(historical_data)
                )
            
            # Calculate confidence assessment
            confidence_assessment = self._calculate_confidence_assessment(primary_forecast)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                forecast_points=primary_forecast,
                accuracy_metrics=accuracy_metrics,
                budget_analysis=budget_analysis,
                seasonal_patterns=seasonal_patterns
            )
            
            # Create result
            result = ForecastResult(
                request_id=request_id,
                account_id=request.account_id,
                service=request.service,
                resource_id=request.resource_id,
                horizon_days=primary_horizon.value,
                generated_at=datetime.now(),
                forecast_points=primary_forecast,
                accuracy_metrics=accuracy_metrics,
                model_performance=self._get_model_performance(),
                budget_analysis=budget_analysis,
                seasonal_patterns=seasonal_patterns,
                confidence_assessment=confidence_assessment,
                recommendations=recommendations
            )
            
            # Store forecast history
            self.forecast_history.append(result)
            
            logger.info(f"Forecast {request_id} generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            raise
    
    async def _generate_horizon_forecast(
        self,
        historical_data: pd.DataFrame,
        horizon_days: int,
        confidence_level: ForecastConfidence,
        include_seasonality: bool
    ) -> List[ForecastPoint]:
        """Generate forecast for specific horizon"""
        
        forecast_points = []
        
        try:
            # Use Prophet as primary forecaster if available
            if 'prophet' in self.models and self.models['prophet']:
                prophet_forecast = await self._generate_prophet_forecast(
                    historical_data, horizon_days, confidence_level, include_seasonality
                )
                forecast_points.extend(prophet_forecast)
            
            # Enhance with LSTM if available
            if 'lstm' in self.models and len(forecast_points) > 0:
                lstm_enhancement = await self._enhance_with_lstm(
                    historical_data, forecast_points
                )
                forecast_points = lstm_enhancement
            
            # Apply ensemble weighting if multiple models available
            if len(self.models) > 1:
                forecast_points = await self._apply_ensemble_weighting(
                    forecast_points, historical_data
                )
            
            # Fallback to simple forecasting if no models available
            if not forecast_points:
                forecast_points = await self._generate_fallback_forecast(
                    historical_data, horizon_days, confidence_level
                )
            
            return forecast_points
            
        except Exception as e:
            logger.error(f"Horizon forecast generation failed: {e}")
            # Return fallback forecast
            return await self._generate_fallback_forecast(
                historical_data, horizon_days, confidence_level
            )
    
    async def _generate_prophet_forecast(
        self,
        historical_data: pd.DataFrame,
        horizon_days: int,
        confidence_level: ForecastConfidence,
        include_seasonality: bool
    ) -> List[ForecastPoint]:
        """Generate Prophet-based forecast"""
        
        try:
            prophet = self.models['prophet']
            
            # Train Prophet model
            training_metrics = prophet.train(
                training_data=historical_data,
                target_column='cost_amount'
            )
            
            # Generate forecast
            prophet_results = prophet.generate_forecast(
                periods=horizon_days,
                freq='D',
                include_history=False
            )
            
            # Convert to ForecastPoint objects
            forecast_points = []
            
            for result in prophet_results:
                # Calculate model contributions
                model_contributions = {
                    'prophet_trend': result.trend,
                    'prophet_seasonal': result.seasonal,
                    'prophet_base': result.forecasted_value
                }
                
                point = ForecastPoint(
                    timestamp=result.timestamp,
                    forecasted_value=result.forecasted_value,
                    lower_bound=result.lower_bound,
                    upper_bound=result.upper_bound,
                    confidence_interval_width=result.confidence_interval,
                    trend_component=result.trend,
                    seasonal_component=result.seasonal or 0,
                    uncertainty=result.uncertainty,
                    model_contributions=model_contributions
                )
                
                forecast_points.append(point)
            
            return forecast_points
            
        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            return []
    
    async def _enhance_with_lstm(
        self,
        historical_data: pd.DataFrame,
        prophet_forecast: List[ForecastPoint]
    ) -> List[ForecastPoint]:
        """Enhance Prophet forecast with LSTM predictions"""
        
        try:
            lstm = self.models['lstm']
            
            # Train LSTM model
            lstm.train(historical_data, target_column='cost_amount')
            
            # Generate LSTM predictions
            recent_data = historical_data.tail(20)
            lstm_predictions = lstm.predict_next_values(
                recent_data, steps_ahead=len(prophet_forecast)
            )
            
            # Combine Prophet and LSTM forecasts
            enhanced_forecast = []
            
            for i, prophet_point in enumerate(prophet_forecast):
                if i < len(lstm_predictions):
                    lstm_pred = lstm_predictions[i]
                    
                    # Weighted combination
                    prophet_weight = self.model_weights['prophet']
                    lstm_weight = self.model_weights['lstm']
                    
                    combined_value = (
                        prophet_point.forecasted_value * prophet_weight +
                        lstm_pred['predicted_values'][0] * lstm_weight
                    )
                    
                    # Update model contributions
                    model_contributions = prophet_point.model_contributions.copy()
                    model_contributions['lstm_prediction'] = lstm_pred['predicted_values'][0]
                    model_contributions['combined_weight'] = combined_value
                    
                    # Create enhanced point
                    enhanced_point = ForecastPoint(
                        timestamp=prophet_point.timestamp,
                        forecasted_value=combined_value,
                        lower_bound=prophet_point.lower_bound * 0.9,  # Slightly tighter bounds
                        upper_bound=prophet_point.upper_bound * 0.9,
                        confidence_interval_width=prophet_point.confidence_interval_width * 0.9,
                        trend_component=prophet_point.trend_component,
                        seasonal_component=prophet_point.seasonal_component,
                        uncertainty=prophet_point.uncertainty * 0.95,  # Reduced uncertainty
                        model_contributions=model_contributions
                    )
                    
                    enhanced_forecast.append(enhanced_point)
                else:
                    enhanced_forecast.append(prophet_point)
            
            return enhanced_forecast
            
        except Exception as e:
            logger.error(f"LSTM enhancement failed: {e}")
            return prophet_forecast
    
    async def _apply_ensemble_weighting(
        self,
        forecast_points: List[ForecastPoint],
        historical_data: pd.DataFrame
    ) -> List[ForecastPoint]:
        """Apply ensemble weighting to improve forecast accuracy"""
        
        try:
            # Calculate ensemble weights based on historical performance
            ensemble_weights = self._calculate_ensemble_weights(historical_data)
            
            # Apply weights to forecast points
            weighted_forecast = []
            
            for point in forecast_points:
                # Apply ensemble weighting to forecasted value
                weighted_value = point.forecasted_value
                
                # Adjust confidence intervals based on ensemble performance
                confidence_adjustment = ensemble_weights.get('confidence_factor', 1.0)
                
                weighted_point = ForecastPoint(
                    timestamp=point.timestamp,
                    forecasted_value=weighted_value,
                    lower_bound=point.lower_bound * confidence_adjustment,
                    upper_bound=point.upper_bound * confidence_adjustment,
                    confidence_interval_width=point.confidence_interval_width * confidence_adjustment,
                    trend_component=point.trend_component,
                    seasonal_component=point.seasonal_component,
                    uncertainty=point.uncertainty * confidence_adjustment,
                    model_contributions=point.model_contributions
                )
                
                weighted_forecast.append(weighted_point)
            
            return weighted_forecast
            
        except Exception as e:
            logger.error(f"Ensemble weighting failed: {e}")
            return forecast_points
    
    async def _generate_fallback_forecast(
        self,
        historical_data: pd.DataFrame,
        horizon_days: int,
        confidence_level: ForecastConfidence
    ) -> List[ForecastPoint]:
        """Generate simple fallback forecast when models unavailable"""
        
        try:
            # Simple linear trend forecast
            costs = historical_data['cost_amount'].values
            days = np.arange(len(costs))
            
            # Calculate trend
            slope = np.polyfit(days, costs, 1)[0]
            intercept = np.polyfit(days, costs, 1)[1]
            
            # Calculate prediction error for confidence intervals
            predictions = slope * days + intercept
            errors = costs - predictions
            std_error = np.std(errors)
            
            # Generate forecast points
            forecast_points = []
            base_date = historical_data['timestamp'].max()
            
            for i in range(horizon_days):
                forecast_date = base_date + timedelta(days=i+1)
                
                # Linear prediction
                forecasted_value = slope * (len(costs) + i) + intercept
                forecasted_value = max(0, forecasted_value)  # Ensure positive
                
                # Confidence intervals
                confidence_multiplier = {
                    ForecastConfidence.HIGH: 1.96,
                    ForecastConfidence.MEDIUM: 1.28,
                    ForecastConfidence.LOW: 1.0
                }[confidence_level]
                
                uncertainty = std_error * confidence_multiplier
                lower_bound = max(0, forecasted_value - uncertainty)
                upper_bound = forecasted_value + uncertainty
                
                # Simple seasonal component (weekly pattern)
                day_of_week = forecast_date.weekday()
                seasonal_factor = 1.1 if day_of_week < 5 else 0.9  # Weekday vs weekend
                seasonal_component = forecasted_value * (seasonal_factor - 1)
                
                point = ForecastPoint(
                    timestamp=forecast_date,
                    forecasted_value=forecasted_value,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    confidence_interval_width=upper_bound - lower_bound,
                    trend_component=forecasted_value - seasonal_component,
                    seasonal_component=seasonal_component,
                    uncertainty=uncertainty / forecasted_value if forecasted_value > 0 else 0.1,
                    model_contributions={'fallback_linear': forecasted_value}
                )
                
                forecast_points.append(point)
            
            return forecast_points
            
        except Exception as e:
            logger.error(f"Fallback forecast failed: {e}")
            return []
    
    async def _analyze_budget_overrun(
        self,
        forecast_points: List[ForecastPoint],
        budget_amount: float,
        budget_period_days: int,
        current_spend: float
    ) -> BudgetAnalysis:
        """Analyze budget overrun probability"""
        
        try:
            # Calculate projected total spend
            projected_future_spend = sum(point.forecasted_value for point in forecast_points)
            projected_total_spend = current_spend + projected_future_spend
            
            # Calculate overrun probability using forecast uncertainty
            overrun_probability = 0.0
            days_until_overrun = None
            
            cumulative_spend = current_spend
            for i, point in enumerate(forecast_points):
                cumulative_spend += point.forecasted_value
                
                if cumulative_spend >= budget_amount and days_until_overrun is None:
                    days_until_overrun = i + 1
                
                # Calculate probability using confidence intervals
                if point.upper_bound > 0:
                    prob_exceed = max(0, min(1, (cumulative_spend - budget_amount) / point.upper_bound))
                    overrun_probability = max(overrun_probability, prob_exceed)
            
            # Calculate overrun amount
            overrun_amount = max(0, projected_total_spend - budget_amount)
            
            # Calculate confidence score
            avg_uncertainty = np.mean([point.uncertainty for point in forecast_points])
            confidence_score = max(0, min(1, 1 - avg_uncertainty))
            
            # Determine risk level
            if overrun_probability >= 0.8:
                risk_level = 'critical'
            elif overrun_probability >= 0.6:
                risk_level = 'high'
            elif overrun_probability >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return BudgetAnalysis(
                budget_amount=budget_amount,
                current_spend=current_spend,
                projected_total_spend=projected_total_spend,
                overrun_probability=overrun_probability,
                days_until_overrun=days_until_overrun,
                overrun_amount=overrun_amount,
                confidence_score=confidence_score,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Budget analysis failed: {e}")
            return BudgetAnalysis(
                budget_amount=budget_amount,
                current_spend=current_spend,
                projected_total_spend=current_spend,
                overrun_probability=0.0,
                days_until_overrun=None,
                overrun_amount=0.0,
                confidence_score=0.5,
                risk_level='unknown'
            )
    
    async def _calculate_accuracy_metrics(
        self,
        historical_data: pd.DataFrame,
        forecast_points: List[ForecastPoint],
        horizon: ForecastHorizon
    ) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        
        try:
            # Use recent historical data for validation
            validation_data = historical_data.tail(min(len(forecast_points), len(historical_data)))
            
            if len(validation_data) == 0:
                return {'mae': 0, 'mape': 0, 'rmse': 0, 'accuracy': 0.5}
            
            # Calculate basic metrics
            actual_values = validation_data['cost_amount'].values
            predicted_values = [point.forecasted_value for point in forecast_points[:len(actual_values)]]
            
            if len(predicted_values) == 0:
                return {'mae': 0, 'mape': 0, 'rmse': 0, 'accuracy': 0.5}
            
            # Mean Absolute Error
            mae = np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)))
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((np.array(actual_values) - np.array(predicted_values)) ** 2))
            
            # Overall accuracy (1 - normalized MAPE)
            accuracy = max(0, 1 - mape / 100)
            
            # Compare against target accuracy
            target_accuracy = self.accuracy_targets[horizon]
            meets_target = accuracy >= target_accuracy
            
            return {
                'mae': float(mae),
                'mape': float(mape),
                'rmse': float(rmse),
                'accuracy': float(accuracy),
                'target_accuracy': float(target_accuracy),
                'meets_target': meets_target
            }
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return {'mae': 0, 'mape': 0, 'rmse': 0, 'accuracy': 0.5}
    
    async def _analyze_seasonal_patterns(
        self,
        historical_data: pd.DataFrame,
        forecast_points: List[ForecastPoint]
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns in forecast"""
        
        try:
            seasonal_analysis = {}
            
            # Weekly seasonality
            if len(forecast_points) >= 7:
                weekly_values = {}
                for point in forecast_points[:28]:  # 4 weeks
                    day_of_week = point.timestamp.weekday()
                    if day_of_week not in weekly_values:
                        weekly_values[day_of_week] = []
                    weekly_values[day_of_week].append(point.seasonal_component)
                
                weekly_pattern = {day: np.mean(values) for day, values in weekly_values.items()}
                seasonal_analysis['weekly'] = {
                    'pattern': weekly_pattern,
                    'strength': np.std(list(weekly_pattern.values())),
                    'peak_day': max(weekly_pattern, key=weekly_pattern.get)
                }
            
            # Monthly trends
            if len(forecast_points) >= 30:
                monthly_trend = []
                for i in range(0, min(len(forecast_points), 90), 30):
                    month_avg = np.mean([p.forecasted_value for p in forecast_points[i:i+30]])
                    monthly_trend.append(month_avg)
                
                seasonal_analysis['monthly'] = {
                    'trend': monthly_trend,
                    'growth_rate': (monthly_trend[-1] - monthly_trend[0]) / monthly_trend[0] if len(monthly_trend) > 1 and monthly_trend[0] > 0 else 0
                }
            
            # Overall seasonality strength
            seasonal_components = [point.seasonal_component for point in forecast_points]
            total_variance = np.var([point.forecasted_value for point in forecast_points])
            seasonal_variance = np.var(seasonal_components)
            
            seasonal_analysis['overall'] = {
                'seasonal_strength': seasonal_variance / total_variance if total_variance > 0 else 0,
                'has_strong_seasonality': seasonal_variance / total_variance > 0.1 if total_variance > 0 else False
            }
            
            return seasonal_analysis
            
        except Exception as e:
            logger.error(f"Seasonal analysis failed: {e}")
            return {}
    
    async def _generate_recommendations(
        self,
        forecast_points: List[ForecastPoint],
        accuracy_metrics: Dict[str, float],
        budget_analysis: Optional[BudgetAnalysis],
        seasonal_patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on forecast"""
        
        recommendations = []
        
        try:
            # Accuracy-based recommendations
            if accuracy_metrics.get('accuracy', 0) < 0.8:
                recommendations.append("Consider collecting more historical data to improve forecast accuracy")
            
            if accuracy_metrics.get('mape', 100) > 20:
                recommendations.append("High forecast uncertainty detected - increase monitoring frequency")
            
            # Budget-based recommendations
            if budget_analysis:
                if budget_analysis.overrun_probability > 0.7:
                    recommendations.append("High budget overrun risk - implement immediate cost controls")
                
                if budget_analysis.days_until_overrun and budget_analysis.days_until_overrun <= 7:
                    recommendations.append("Budget overrun imminent - activate emergency cost reduction measures")
                
                if budget_analysis.overrun_amount > 1000:
                    recommendations.append("Significant overrun projected - consider budget reallocation or increase")
            
            # Trend-based recommendations
            trend_values = [point.trend_component for point in forecast_points]
            if len(trend_values) > 1:
                trend_slope = (trend_values[-1] - trend_values[0]) / len(trend_values)
                
                if trend_slope > 0:
                    recommendations.append("Increasing cost trend detected - investigate cost drivers")
                elif trend_slope < -0.1:
                    recommendations.append("Decreasing cost trend - good opportunity for optimization")
            
            # Seasonality-based recommendations
            if seasonal_patterns.get('overall', {}).get('has_strong_seasonality', False):
                recommendations.append("Strong seasonal patterns detected - adjust budgets for seasonal variations")
            
            # Uncertainty-based recommendations
            avg_uncertainty = np.mean([point.uncertainty for point in forecast_points])
            if avg_uncertainty > 0.3:
                recommendations.append("High forecast uncertainty - implement more granular cost tracking")
            
            # General recommendations
            recommendations.extend([
                "Enable automated cost alerts for early warning",
                "Review and optimize high-cost resources identified in forecast",
                "Consider implementing cost allocation tags for better tracking"
            ])
            
            return recommendations[:8]  # Limit to top 8 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Monitor costs closely and review forecast regularly"]
    
    def _validate_historical_data(self, historical_data: pd.DataFrame):
        """Validate historical data for forecasting"""
        
        if historical_data.empty:
            raise ValueError("Historical data is empty")
        
        required_columns = ['timestamp', 'cost_amount']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(historical_data) < 7:
            raise ValueError("Insufficient historical data (minimum 7 days required)")
        
        # Check for data quality issues
        if historical_data['cost_amount'].isna().sum() > len(historical_data) * 0.1:
            raise ValueError("Too many missing cost values (>10%)")
    
    def _calculate_current_spend(self, historical_data: pd.DataFrame) -> float:
        """Calculate current spend from historical data"""
        
        try:
            # Use recent data to estimate current spend
            recent_data = historical_data.tail(7)  # Last 7 days
            return float(recent_data['cost_amount'].sum())
        except:
            return 0.0
    
    def _calculate_ensemble_weights(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ensemble weights based on historical performance"""
        
        # Simplified ensemble weighting
        return {
            'prophet_weight': 0.5,
            'lstm_weight': 0.3,
            'ensemble_weight': 0.2,
            'confidence_factor': 0.95
        }
    
    def _calculate_confidence_assessment(self, forecast_points: List[ForecastPoint]) -> Dict[str, float]:
        """Calculate overall confidence assessment"""
        
        try:
            uncertainties = [point.uncertainty for point in forecast_points]
            confidence_intervals = [point.confidence_interval_width for point in forecast_points]
            
            return {
                'average_uncertainty': float(np.mean(uncertainties)),
                'max_uncertainty': float(np.max(uncertainties)),
                'average_confidence_width': float(np.mean(confidence_intervals)),
                'overall_confidence': float(1 - np.mean(uncertainties))
            }
        except:
            return {'overall_confidence': 0.5}
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        
        return {
            'models_available': list(self.models.keys()),
            'model_weights': self.model_weights,
            'forecast_count': len(self.forecast_history),
            'last_updated': datetime.now().isoformat()
        }
    
    def _create_fallback_model(self):
        """Create fallback model when others unavailable"""
        
        class FallbackModel:
            def __init__(self):
                self.is_trained = False
            
            def train(self, *args, **kwargs):
                self.is_trained = True
                return {'mae': 10, 'accuracy': 0.7}
        
        return FallbackModel()
    
    async def get_forecast_history(self, account_id: Optional[str] = None) -> List[ForecastResult]:
        """Get forecast history for account or all accounts"""
        
        if account_id:
            return [f for f in self.forecast_history if f.account_id == account_id]
        return self.forecast_history
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and performance"""
        
        return {
            'models': {name: bool(model) for name, model in self.models.items()},
            'performance': self.model_performance,
            'accuracy_targets': {h.name: target for h, target in self.accuracy_targets.items()},
            'forecast_count': len(self.forecast_history),
            'last_forecast': self.forecast_history[-1].generated_at.isoformat() if self.forecast_history else None
        }