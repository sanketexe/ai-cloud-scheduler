# performance_analyzer.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from performance_monitor import MetricType, MetricsData, CloudResource, ScalingDirection, ResourceCapacity, ScalingRecommendation
from enhanced_models import SeverityLevel


class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class PerformanceTrends:
    """Analysis of performance trends over time"""
    resource_id: str
    analysis_period_start: datetime
    analysis_period_end: datetime
    cpu_trend: TrendDirection
    memory_trend: TrendDirection
    io_trend: TrendDirection
    response_time_trend: TrendDirection
    capacity_utilization_forecast: Dict[str, float]  # Next 7, 30, 90 days
    seasonal_patterns: Dict[str, Any] = field(default_factory=dict)
    growth_rate: float = 0.0  # Percentage growth per month
    confidence_score: float = 0.0  # 0-1, confidence in the analysis


@dataclass
class CapacityForecast:
    """Capacity planning forecast"""
    resource_id: str
    forecast_period_days: int
    current_utilization: Dict[MetricType, float]
    predicted_utilization: Dict[MetricType, List[Tuple[datetime, float]]]
    capacity_exhaustion_dates: Dict[MetricType, Optional[datetime]]
    recommended_actions: List[str]
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceBaseline:
    """Performance baseline for a resource"""
    resource_id: str
    metric_type: MetricType
    baseline_value: float
    baseline_std: float
    percentile_95: float
    percentile_99: float
    sample_size: int
    baseline_period_start: datetime
    baseline_period_end: datetime
    seasonal_adjustments: Dict[str, float] = field(default_factory=dict)


class PerformanceAnalyzer:
    """Analyzes performance trends and provides capacity planning insights"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceAnalyzer")
        self.baselines: Dict[str, Dict[MetricType, PerformanceBaseline]] = {}
        self.trend_history: List[PerformanceTrends] = []
        self.capacity_forecasts: Dict[str, CapacityForecast] = {}
    
    def analyze_performance_trends(self, metrics_data: Dict[str, MetricsData],
                                 analysis_period_days: int = 30) -> Dict[str, PerformanceTrends]:
        """Analyze performance trends for all resources"""
        self.logger.info(f"Analyzing performance trends for {len(metrics_data)} resources")
        
        trends = {}
        
        for resource_id, data in metrics_data.items():
            trend = self._analyze_resource_trends(resource_id, data, analysis_period_days)
            if trend:
                trends[resource_id] = trend
                self.trend_history.append(trend)
        
        # Keep history manageable
        if len(self.trend_history) > 1000:
            self.trend_history = self.trend_history[-500:]
        
        self.logger.info(f"Completed trend analysis for {len(trends)} resources")
        return trends
    
    def _analyze_resource_trends(self, resource_id: str, data: MetricsData,
                               analysis_period_days: int) -> Optional[PerformanceTrends]:
        """Analyze trends for a specific resource"""
        if not data.metrics:
            return None
        
        # Filter data to analysis period
        end_time = data.collection_period_end
        start_time = end_time - timedelta(days=analysis_period_days)
        
        # Analyze trends for each metric type
        cpu_trend = self._analyze_metric_trend(
            data.metrics.get(MetricType.CPU_UTILIZATION, []), start_time, end_time
        )
        memory_trend = self._analyze_metric_trend(
            data.metrics.get(MetricType.MEMORY_UTILIZATION, []), start_time, end_time
        )
        io_trend = self._analyze_metric_trend(
            data.metrics.get(MetricType.DISK_IO, []), start_time, end_time
        )
        response_time_trend = self._analyze_metric_trend(
            data.metrics.get(MetricType.RESPONSE_TIME, []), start_time, end_time
        )
        
        # Calculate capacity utilization forecast
        capacity_forecast = self._forecast_capacity_utilization(resource_id, data)
        
        # Detect seasonal patterns
        seasonal_patterns = self._detect_seasonal_patterns(data)
        
        # Calculate overall growth rate
        growth_rate = self._calculate_growth_rate(data)
        
        # Calculate confidence score
        confidence_score = self._calculate_trend_confidence(data)
        
        return PerformanceTrends(
            resource_id=resource_id,
            analysis_period_start=start_time,
            analysis_period_end=end_time,
            cpu_trend=cpu_trend,
            memory_trend=memory_trend,
            io_trend=io_trend,
            response_time_trend=response_time_trend,
            capacity_utilization_forecast=capacity_forecast,
            seasonal_patterns=seasonal_patterns,
            growth_rate=growth_rate,
            confidence_score=confidence_score
        )
    
    def _analyze_metric_trend(self, time_series: List[Tuple[datetime, float]],
                            start_time: datetime, end_time: datetime) -> TrendDirection:
        """Analyze trend direction for a specific metric"""
        if not time_series or len(time_series) < 10:
            return TrendDirection.STABLE
        
        # Filter to analysis period
        filtered_series = [
            (timestamp, value) for timestamp, value in time_series
            if start_time <= timestamp <= end_time
        ]
        
        if len(filtered_series) < 10:
            return TrendDirection.STABLE
        
        values = [value for _, value in filtered_series]
        
        # Calculate trend using linear regression if available
        if SKLEARN_AVAILABLE and len(values) >= 20:
            return self._calculate_ml_trend(values)
        else:
            return self._calculate_statistical_trend(values)
    
    def _calculate_ml_trend(self, values: List[float]) -> TrendDirection:
        """Calculate trend using machine learning (linear regression)"""
        try:
            X = np.array(range(len(values))).reshape(-1, 1)
            y = np.array(values)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Get slope (coefficient)
            slope = model.coef_[0]
            
            # Calculate R-squared for trend strength
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Determine trend direction
            if r_squared < 0.1:  # Low correlation, likely volatile
                return TrendDirection.VOLATILE
            elif abs(slope) < np.std(values) * 0.01:  # Very small slope
                return TrendDirection.STABLE
            elif slope > 0:
                return TrendDirection.INCREASING
            else:
                return TrendDirection.DECREASING
                
        except Exception as e:
            self.logger.error(f"Error in ML trend calculation: {e}")
            return self._calculate_statistical_trend(values)
    
    def _calculate_statistical_trend(self, values: List[float]) -> TrendDirection:
        """Calculate trend using statistical methods"""
        if len(values) < 10:
            return TrendDirection.STABLE
        
        # Split into first and second half
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]
        
        first_mean = statistics.mean(first_half)
        second_mean = statistics.mean(second_half)
        overall_std = statistics.stdev(values) if len(values) > 1 else 0
        
        # Calculate coefficient of variation to detect volatility
        overall_mean = statistics.mean(values)
        cv = (overall_std / overall_mean) if overall_mean > 0 else 0
        
        if cv > 0.3:  # High coefficient of variation indicates volatility
            return TrendDirection.VOLATILE
        
        # Compare means to determine trend
        difference = second_mean - first_mean
        threshold = overall_std * 0.5  # Threshold for significant change
        
        if abs(difference) < threshold:
            return TrendDirection.STABLE
        elif difference > 0:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING
    
    def _forecast_capacity_utilization(self, resource_id: str, 
                                     data: MetricsData) -> Dict[str, float]:
        """Forecast capacity utilization for next 7, 30, 90 days"""
        forecast = {}
        
        # Focus on CPU and Memory utilization for capacity planning
        key_metrics = [MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION]
        
        for metric_type in key_metrics:
            if metric_type not in data.metrics:
                continue
            
            time_series = data.metrics[metric_type]
            if len(time_series) < 20:  # Need sufficient data for forecasting
                continue
            
            # Forecast for different time horizons
            for days in [7, 30, 90]:
                forecasted_value = self._forecast_metric_value(time_series, days)
                if forecasted_value is not None:
                    key = f"{metric_type.value}_{days}d"
                    forecast[key] = min(100.0, max(0.0, forecasted_value))  # Clamp to 0-100%
        
        return forecast
    
    def _forecast_metric_value(self, time_series: List[Tuple[datetime, float]], 
                             forecast_days: int) -> Optional[float]:
        """Forecast metric value for specified number of days ahead"""
        if len(time_series) < 10:
            return None
        
        values = [value for _, value in time_series]
        
        if SKLEARN_AVAILABLE and len(values) >= 30:
            return self._ml_forecast(values, forecast_days)
        else:
            return self._statistical_forecast(values, forecast_days)
    
    def _ml_forecast(self, values: List[float], forecast_days: int) -> Optional[float]:
        """Machine learning-based forecasting"""
        try:
            # Use polynomial features for better trend capture
            X = np.array(range(len(values))).reshape(-1, 1)
            y = np.array(values)
            
            # Try polynomial regression (degree 2)
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Forecast future value
            future_x = np.array([[len(values) + forecast_days]])
            future_x_poly = poly_features.transform(future_x)
            forecasted_value = model.predict(future_x_poly)[0]
            
            return forecasted_value
            
        except Exception as e:
            self.logger.error(f"Error in ML forecasting: {e}")
            return self._statistical_forecast(values, forecast_days)
    
    def _statistical_forecast(self, values: List[float], forecast_days: int) -> Optional[float]:
        """Statistical forecasting using trend extrapolation"""
        if len(values) < 5:
            return None
        
        # Simple linear trend extrapolation
        recent_values = values[-min(30, len(values)):]  # Use last 30 data points
        
        # Calculate trend slope
        n = len(recent_values)
        x_values = list(range(n))
        
        # Linear regression manually
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(recent_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return recent_values[-1]  # No trend, return last value
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Forecast
        future_x = n + forecast_days
        forecasted_value = slope * future_x + intercept
        
        return forecasted_value
    
    def _detect_seasonal_patterns(self, data: MetricsData) -> Dict[str, Any]:
        """Detect seasonal patterns in the data"""
        patterns = {}
        
        # Analyze daily patterns (hourly variations)
        daily_patterns = self._analyze_daily_patterns(data)
        if daily_patterns:
            patterns['daily'] = daily_patterns
        
        # Analyze weekly patterns (day-of-week variations)
        weekly_patterns = self._analyze_weekly_patterns(data)
        if weekly_patterns:
            patterns['weekly'] = weekly_patterns
        
        return patterns
    
    def _analyze_daily_patterns(self, data: MetricsData) -> Optional[Dict[str, float]]:
        """Analyze daily (hourly) patterns"""
        # Focus on CPU utilization for daily pattern analysis
        if MetricType.CPU_UTILIZATION not in data.metrics:
            return None
        
        time_series = data.metrics[MetricType.CPU_UTILIZATION]
        if len(time_series) < 48:  # Need at least 2 days of data
            return None
        
        # Group by hour of day
        hourly_values = {}
        for timestamp, value in time_series:
            hour = timestamp.hour
            if hour not in hourly_values:
                hourly_values[hour] = []
            hourly_values[hour].append(value)
        
        # Calculate average for each hour
        hourly_averages = {}
        for hour, values in hourly_values.items():
            if len(values) >= 2:  # Need multiple samples
                hourly_averages[hour] = statistics.mean(values)
        
        if len(hourly_averages) < 12:  # Need data for at least half the day
            return None
        
        return hourly_averages
    
    def _analyze_weekly_patterns(self, data: MetricsData) -> Optional[Dict[str, float]]:
        """Analyze weekly (day-of-week) patterns"""
        if MetricType.CPU_UTILIZATION not in data.metrics:
            return None
        
        time_series = data.metrics[MetricType.CPU_UTILIZATION]
        if len(time_series) < 168:  # Need at least 1 week of hourly data
            return None
        
        # Group by day of week (0=Monday, 6=Sunday)
        daily_values = {}
        for timestamp, value in time_series:
            day_of_week = timestamp.weekday()
            if day_of_week not in daily_values:
                daily_values[day_of_week] = []
            daily_values[day_of_week].append(value)
        
        # Calculate average for each day
        daily_averages = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day, values in daily_values.items():
            if len(values) >= 12:  # Need at least 12 hours of data per day
                daily_averages[day_names[day]] = statistics.mean(values)
        
        if len(daily_averages) < 5:  # Need data for at least 5 days
            return None
        
        return daily_averages
    
    def _calculate_growth_rate(self, data: MetricsData) -> float:
        """Calculate overall growth rate across key metrics"""
        growth_rates = []
        
        key_metrics = [MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION]
        
        for metric_type in key_metrics:
            if metric_type not in data.metrics:
                continue
            
            time_series = data.metrics[metric_type]
            if len(time_series) < 30:  # Need at least 30 data points
                continue
            
            values = [value for _, value in time_series]
            
            # Calculate growth rate between first and last quarters
            quarter_size = len(values) // 4
            if quarter_size < 5:
                continue
            
            first_quarter = values[:quarter_size]
            last_quarter = values[-quarter_size:]
            
            first_avg = statistics.mean(first_quarter)
            last_avg = statistics.mean(last_quarter)
            
            if first_avg > 0:
                growth_rate = ((last_avg - first_avg) / first_avg) * 100
                growth_rates.append(growth_rate)
        
        return statistics.mean(growth_rates) if growth_rates else 0.0
    
    def _calculate_trend_confidence(self, data: MetricsData) -> float:
        """Calculate confidence score for trend analysis"""
        confidence_factors = []
        
        # Data quantity factor
        total_data_points = sum(len(ts) for ts in data.metrics.values())
        data_quantity_score = min(1.0, total_data_points / 1000)  # Normalize to 1000 points
        confidence_factors.append(data_quantity_score)
        
        # Data completeness factor (how many key metrics we have)
        key_metrics = [MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION, 
                      MetricType.RESPONSE_TIME, MetricType.THROUGHPUT]
        available_metrics = len([m for m in key_metrics if m in data.metrics])
        completeness_score = available_metrics / len(key_metrics)
        confidence_factors.append(completeness_score)
        
        # Time span factor
        if data.metrics:
            time_span = (data.collection_period_end - data.collection_period_start).total_seconds()
            time_span_days = time_span / (24 * 3600)
            time_span_score = min(1.0, time_span_days / 30)  # Normalize to 30 days
            confidence_factors.append(time_span_score)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.0
    
    def generate_scaling_recommendations(self, trends: Dict[str, PerformanceTrends],
                                       current_capacities: Dict[str, ResourceCapacity]) -> List[ScalingRecommendation]:
        """Generate scaling recommendations based on performance trends"""
        recommendations = []
        
        for resource_id, trend in trends.items():
            current_capacity = current_capacities.get(resource_id)
            if not current_capacity:
                continue
            
            recommendation = self._generate_resource_scaling_recommendation(
                resource_id, trend, current_capacity
            )
            
            if recommendation:
                recommendations.append(recommendation)
        
        self.logger.info(f"Generated {len(recommendations)} scaling recommendations")
        return recommendations
    
    def _generate_resource_scaling_recommendation(self, resource_id: str,
                                                trend: PerformanceTrends,
                                                current_capacity: ResourceCapacity) -> Optional[ScalingRecommendation]:
        """Generate scaling recommendation for a specific resource"""
        # Analyze CPU trend and forecast
        cpu_forecast_30d = trend.capacity_utilization_forecast.get('cpu_utilization_30d', 0)
        memory_forecast_30d = trend.capacity_utilization_forecast.get('memory_utilization_30d', 0)
        
        # Determine if scaling is needed
        scaling_needed = False
        scaling_direction = ScalingDirection.NONE
        rationale_parts = []
        urgency = SeverityLevel.LOW
        
        # Check CPU scaling needs
        if cpu_forecast_30d > 85:
            scaling_needed = True
            scaling_direction = ScalingDirection.UP
            rationale_parts.append(f"CPU utilization forecasted to reach {cpu_forecast_30d:.1f}% in 30 days")
            if cpu_forecast_30d > 95:
                urgency = SeverityLevel.HIGH
            elif cpu_forecast_30d > 90:
                urgency = SeverityLevel.MEDIUM
        
        # Check memory scaling needs
        if memory_forecast_30d > 85:
            scaling_needed = True
            scaling_direction = ScalingDirection.UP
            rationale_parts.append(f"Memory utilization forecasted to reach {memory_forecast_30d:.1f}% in 30 days")
            if memory_forecast_30d > 95:
                urgency = max(urgency, SeverityLevel.HIGH, key=lambda x: x.value)
            elif memory_forecast_30d > 90:
                urgency = max(urgency, SeverityLevel.MEDIUM, key=lambda x: x.value)
        
        # Check for scale-down opportunities
        if (not scaling_needed and 
            cpu_forecast_30d < 30 and memory_forecast_30d < 30 and
            trend.cpu_trend == TrendDirection.STABLE and trend.memory_trend == TrendDirection.STABLE):
            scaling_needed = True
            scaling_direction = ScalingDirection.DOWN
            rationale_parts.append("Low utilization with stable trends suggests over-provisioning")
            urgency = SeverityLevel.LOW
        
        if not scaling_needed:
            return None
        
        # Calculate recommended capacity
        recommended_capacity = self._calculate_recommended_capacity(
            current_capacity, scaling_direction, cpu_forecast_30d, memory_forecast_30d
        )
        
        # Estimate cost impact (simplified)
        cost_impact = self._estimate_cost_impact(current_capacity, recommended_capacity)
        
        # Calculate confidence score
        confidence_score = min(1.0, trend.confidence_score + 0.2)  # Boost confidence slightly
        
        rationale = "; ".join(rationale_parts)
        
        return ScalingRecommendation(
            resource_id=resource_id,
            current_capacity=current_capacity,
            recommended_capacity=recommended_capacity,
            scaling_direction=scaling_direction,
            confidence_score=confidence_score,
            estimated_cost_impact=cost_impact,
            rationale=rationale,
            urgency=urgency
        )
    
    def _calculate_recommended_capacity(self, current: ResourceCapacity,
                                      direction: ScalingDirection,
                                      cpu_forecast: float, memory_forecast: float) -> ResourceCapacity:
        """Calculate recommended resource capacity"""
        if direction == ScalingDirection.UP:
            # Scale up based on forecasted utilization
            cpu_multiplier = max(1.2, (cpu_forecast + 20) / 70)  # Target 70% utilization
            memory_multiplier = max(1.2, (memory_forecast + 20) / 70)
            
            # Use the higher multiplier but cap at 2x
            multiplier = min(2.0, max(cpu_multiplier, memory_multiplier))
            
            return ResourceCapacity(
                cpu_cores=max(current.cpu_cores + 1, int(current.cpu_cores * multiplier)),
                memory_gb=max(current.memory_gb + 2, current.memory_gb * multiplier),
                disk_gb=current.disk_gb,  # Keep disk same for now
                network_bandwidth_mbps=current.network_bandwidth_mbps * multiplier,
                instance_type=f"{current.instance_type}_scaled_up"
            )
        
        elif direction == ScalingDirection.DOWN:
            # Scale down conservatively
            multiplier = 0.8  # Reduce by 20%
            
            return ResourceCapacity(
                cpu_cores=max(1, int(current.cpu_cores * multiplier)),
                memory_gb=max(1.0, current.memory_gb * multiplier),
                disk_gb=current.disk_gb,
                network_bandwidth_mbps=current.network_bandwidth_mbps * multiplier,
                instance_type=f"{current.instance_type}_scaled_down"
            )
        
        else:
            return current
    
    def _estimate_cost_impact(self, current: ResourceCapacity, 
                            recommended: ResourceCapacity) -> float:
        """Estimate cost impact of scaling recommendation"""
        # Simplified cost calculation based on CPU and memory changes
        cpu_cost_per_core = 50.0  # $50/month per CPU core
        memory_cost_per_gb = 10.0  # $10/month per GB RAM
        
        current_cost = (current.cpu_cores * cpu_cost_per_core + 
                       current.memory_gb * memory_cost_per_gb)
        
        recommended_cost = (recommended.cpu_cores * cpu_cost_per_core + 
                          recommended.memory_gb * memory_cost_per_gb)
        
        return recommended_cost - current_cost
    
    def create_capacity_forecast(self, resource_id: str, trends: PerformanceTrends,
                               forecast_days: int = 90) -> CapacityForecast:
        """Create detailed capacity forecast for a resource"""
        # Get current utilization from trends
        current_utilization = {}
        predicted_utilization = {}
        capacity_exhaustion_dates = {}
        
        key_metrics = [MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION]
        
        for metric_type in key_metrics:
            # Extract current utilization
            forecast_key = f"{metric_type.value}_{forecast_days}d"
            if forecast_key in trends.capacity_utilization_forecast:
                current_util = trends.capacity_utilization_forecast.get(f"{metric_type.value}_7d", 50.0)
                future_util = trends.capacity_utilization_forecast[forecast_key]
                
                current_utilization[metric_type] = current_util
                
                # Generate prediction timeline
                prediction_timeline = []
                for day in range(0, forecast_days + 1, 7):  # Weekly intervals
                    date = datetime.now() + timedelta(days=day)
                    # Linear interpolation between current and forecasted
                    progress = day / forecast_days if forecast_days > 0 else 0
                    predicted_value = current_util + (future_util - current_util) * progress
                    prediction_timeline.append((date, predicted_value))
                
                predicted_utilization[metric_type] = prediction_timeline
                
                # Calculate capacity exhaustion date (when utilization reaches 95%)
                if future_util > current_util and future_util > 95:
                    # Linear extrapolation to find when 95% is reached
                    days_to_95 = ((95 - current_util) / (future_util - current_util)) * forecast_days
                    if days_to_95 > 0:
                        exhaustion_date = datetime.now() + timedelta(days=days_to_95)
                        capacity_exhaustion_dates[metric_type] = exhaustion_date
        
        # Generate recommendations
        recommendations = []
        if any(date for date in capacity_exhaustion_dates.values() if date and date < datetime.now() + timedelta(days=30)):
            recommendations.append("Immediate scaling required - capacity exhaustion predicted within 30 days")
        elif any(date for date in capacity_exhaustion_dates.values() if date and date < datetime.now() + timedelta(days=60)):
            recommendations.append("Plan scaling within next 30 days - capacity exhaustion predicted within 60 days")
        else:
            recommendations.append("Monitor trends - no immediate scaling required")
        
        return CapacityForecast(
            resource_id=resource_id,
            forecast_period_days=forecast_days,
            current_utilization=current_utilization,
            predicted_utilization=predicted_utilization,
            capacity_exhaustion_dates=capacity_exhaustion_dates,
            recommended_actions=recommendations,
            confidence_score=trends.confidence_score
        )