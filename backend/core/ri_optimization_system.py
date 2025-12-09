"""
Reserved Instance and Commitment Optimization System

This module provides comprehensive RI optimization capabilities including:
- Usage pattern analysis for stable workloads
- Intelligent RI recommendation engine
- Savings and ROI calculations
- RI utilization tracking
- Portfolio management and optimization
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class InstanceType(Enum):
    """Instance types for RI optimization"""
    COMPUTE = "compute"
    DATABASE = "database"
    STORAGE = "storage"


class CommitmentTerm(Enum):
    """RI commitment terms"""
    ONE_YEAR = "1_year"
    THREE_YEAR = "3_year"


class PaymentOption(Enum):
    """RI payment options"""
    NO_UPFRONT = "no_upfront"
    PARTIAL_UPFRONT = "partial_upfront"
    ALL_UPFRONT = "all_upfront"


@dataclass
class UsageDataPoint:
    """Individual usage measurement"""
    timestamp: datetime
    resource_id: str
    instance_type: str
    region: str
    usage_hours: float
    cost: float


@dataclass
class UsagePattern:
    """Analyzed usage pattern for a resource"""
    resource_id: str
    instance_type: str
    region: str
    average_usage: float
    peak_usage: float
    minimum_usage: float
    stability_score: float
    predictability_score: float
    seasonal_patterns: Dict[str, float]
    trend_direction: str
    confidence_level: float


@dataclass
class SeasonalComponent:
    """Seasonal usage component"""
    pattern_type: str  # daily, weekly, monthly, yearly
    amplitude: float
    phase: float
    confidence: float


class UsageAnalyzer:
    """
    Analyzes historical usage patterns to identify stable workloads
    suitable for reserved instance purchases.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".UsageAnalyzer")
        self.min_analysis_days = 30
        self.stability_threshold = 0.7
        self.predictability_threshold = 0.8
    
    def analyze_usage_patterns(self, usage_data: List[UsageDataPoint], 
                             analysis_period_days: int = 90) -> List[UsagePattern]:
        """
        Analyze historical usage data to identify patterns and stability.
        
        Args:
            usage_data: Historical usage data points
            analysis_period_days: Period for analysis (default 90 days)
            
        Returns:
            List of usage patterns for each resource
        """
        self.logger.info(f"Analyzing usage patterns for {len(usage_data)} data points")
        
        if analysis_period_days < self.min_analysis_days:
            raise ValueError(f"Analysis period must be at least {self.min_analysis_days} days")
        
        # Group usage data by resource
        resource_usage = self._group_usage_by_resource(usage_data)
        
        patterns = []
        for resource_id, resource_data in resource_usage.items():
            if len(resource_data) < self.min_analysis_days:
                self.logger.warning(f"Insufficient data for resource {resource_id}")
                continue
                
            pattern = self._analyze_resource_pattern(resource_id, resource_data)
            if pattern:
                patterns.append(pattern)
        
        self.logger.info(f"Generated {len(patterns)} usage patterns")
        return patterns
    
    def calculate_workload_stability(self, usage_data: List[UsageDataPoint]) -> float:
        """
        Calculate stability score based on usage variance.
        
        Args:
            usage_data: Usage data points for analysis
            
        Returns:
            Stability score (0.0 to 1.0, higher is more stable)
        """
        if len(usage_data) < 2:
            return 0.0
        
        usage_values = [point.usage_hours for point in usage_data]
        mean_usage = statistics.mean(usage_values)
        
        if mean_usage == 0:
            return 0.0
        
        # Calculate coefficient of variation (lower is more stable)
        std_dev = statistics.stdev(usage_values)
        cv = std_dev / mean_usage
        
        # Convert to stability score (inverse relationship)
        stability_score = max(0.0, 1.0 - min(cv, 1.0))
        
        return stability_score
    
    def calculate_predictability_score(self, usage_data: List[UsageDataPoint]) -> float:
        """
        Calculate predictability score based on pattern consistency.
        
        Args:
            usage_data: Usage data points for analysis
            
        Returns:
            Predictability score (0.0 to 1.0, higher is more predictable)
        """
        if len(usage_data) < 7:  # Need at least a week of data
            return 0.0
        
        # Convert to time series
        df = pd.DataFrame([
            {'timestamp': point.timestamp, 'usage': point.usage_hours}
            for point in usage_data
        ])
        df = df.sort_values('timestamp')
        
        # Calculate autocorrelation for different lags
        usage_series = df['usage']
        autocorrelations = []
        
        for lag in [1, 7, 24]:  # 1 hour, 1 day, 1 week lags
            if len(usage_series) > lag:
                corr = usage_series.autocorr(lag=lag)
                if not pd.isna(corr):
                    autocorrelations.append(abs(corr))
        
        if not autocorrelations:
            return 0.0
        
        # Average autocorrelation as predictability measure
        predictability = statistics.mean(autocorrelations)
        return min(predictability, 1.0)
    
    def identify_seasonal_patterns(self, usage_data: List[UsageDataPoint]) -> Dict[str, SeasonalComponent]:
        """
        Identify seasonal patterns in usage data.
        
        Args:
            usage_data: Usage data points for analysis
            
        Returns:
            Dictionary of seasonal components by pattern type
        """
        if len(usage_data) < 168:  # Need at least a week of hourly data
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'timestamp': point.timestamp,
                'usage': point.usage_hours,
                'hour': point.timestamp.hour,
                'day_of_week': point.timestamp.weekday(),
                'day_of_month': point.timestamp.day
            }
            for point in usage_data
        ])
        
        seasonal_patterns = {}
        
        # Daily pattern (hourly variation)
        if len(df) >= 24:
            hourly_avg = df.groupby('hour')['usage'].mean()
            daily_amplitude = (hourly_avg.max() - hourly_avg.min()) / hourly_avg.mean() if hourly_avg.mean() > 0 else 0
            
            seasonal_patterns['daily'] = SeasonalComponent(
                pattern_type='daily',
                amplitude=daily_amplitude,
                phase=hourly_avg.idxmax(),  # Peak hour
                confidence=min(1.0, len(df) / 168)  # Confidence based on data availability
            )
        
        # Weekly pattern (day-of-week variation)
        if len(df) >= 168:  # At least a week
            weekly_avg = df.groupby('day_of_week')['usage'].mean()
            weekly_amplitude = (weekly_avg.max() - weekly_avg.min()) / weekly_avg.mean() if weekly_avg.mean() > 0 else 0
            
            seasonal_patterns['weekly'] = SeasonalComponent(
                pattern_type='weekly',
                amplitude=weekly_amplitude,
                phase=weekly_avg.idxmax(),  # Peak day
                confidence=min(1.0, len(df) / (168 * 4))  # Confidence based on weeks of data
            )
        
        return seasonal_patterns
    
    def forecast_usage_trend(self, usage_data: List[UsageDataPoint], 
                           forecast_days: int = 30) -> Tuple[str, float]:
        """
        Forecast usage trend direction and magnitude.
        
        Args:
            usage_data: Historical usage data
            forecast_days: Number of days to forecast
            
        Returns:
            Tuple of (trend_direction, confidence_level)
        """
        if len(usage_data) < 14:  # Need at least 2 weeks
            return "stable", 0.0
        
        # Convert to time series
        df = pd.DataFrame([
            {'timestamp': point.timestamp, 'usage': point.usage_hours}
            for point in usage_data
        ])
        df = df.sort_values('timestamp')
        
        # Calculate daily averages
        df['date'] = df['timestamp'].dt.date
        daily_usage = df.groupby('date')['usage'].mean().reset_index()
        daily_usage['day_number'] = range(len(daily_usage))
        
        # Simple linear regression for trend
        if len(daily_usage) >= 7:
            x = daily_usage['day_number'].values
            y = daily_usage['usage'].values
            
            # Calculate slope
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            
            # Determine trend direction
            mean_usage = np.mean(y)
            relative_slope = slope / mean_usage if mean_usage > 0 else 0
            
            if abs(relative_slope) < 0.01:  # Less than 1% change per day
                trend_direction = "stable"
            elif relative_slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # Calculate R-squared for confidence
            y_pred = slope * x + (np.mean(y) - slope * np.mean(x))
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            confidence = max(0.0, min(1.0, r_squared))
            
            return trend_direction, confidence
        
        return "stable", 0.0
    
    def _group_usage_by_resource(self, usage_data: List[UsageDataPoint]) -> Dict[str, List[UsageDataPoint]]:
        """Group usage data by resource ID"""
        resource_usage = defaultdict(list)
        for point in usage_data:
            resource_usage[point.resource_id].append(point)
        return dict(resource_usage)
    
    def _analyze_resource_pattern(self, resource_id: str, 
                                resource_data: List[UsageDataPoint]) -> Optional[UsagePattern]:
        """Analyze usage pattern for a single resource"""
        try:
            # Sort by timestamp
            resource_data.sort(key=lambda x: x.timestamp)
            
            # Calculate basic statistics
            usage_values = [point.usage_hours for point in resource_data]
            average_usage = statistics.mean(usage_values)
            peak_usage = max(usage_values)
            minimum_usage = min(usage_values)
            
            # Calculate stability and predictability
            stability_score = self.calculate_workload_stability(resource_data)
            predictability_score = self.calculate_predictability_score(resource_data)
            
            # Identify seasonal patterns
            seasonal_patterns = self.identify_seasonal_patterns(resource_data)
            
            # Forecast trend
            trend_direction, confidence_level = self.forecast_usage_trend(resource_data)
            
            # Get resource metadata
            instance_type = resource_data[0].instance_type
            region = resource_data[0].region
            
            return UsagePattern(
                resource_id=resource_id,
                instance_type=instance_type,
                region=region,
                average_usage=average_usage,
                peak_usage=peak_usage,
                minimum_usage=minimum_usage,
                stability_score=stability_score,
                predictability_score=predictability_score,
                seasonal_patterns={k: v.amplitude for k, v in seasonal_patterns.items()},
                trend_direction=trend_direction,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern for resource {resource_id}: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    analyzer = UsageAnalyzer()
    
    # Generate sample usage data
    sample_data = []
    base_time = datetime.now() - timedelta(days=90)
    
    for i in range(90 * 24):  # 90 days of hourly data
        timestamp = base_time + timedelta(hours=i)
        
        # Simulate stable workload with daily pattern
        base_usage = 8.0  # 8 hours base usage
        daily_variation = 2.0 * np.sin(2 * np.pi * timestamp.hour / 24)
        noise = np.random.normal(0, 0.5)
        
        usage_hours = max(0, base_usage + daily_variation + noise)
        
        sample_data.append(UsageDataPoint(
            timestamp=timestamp,
            resource_id="i-1234567890abcdef0",
            instance_type="m5.large",
            region="us-east-1",
            usage_hours=usage_hours,
            cost=usage_hours * 0.096  # $0.096 per hour
        ))
    
    # Analyze patterns
    patterns = analyzer.analyze_usage_patterns(sample_data)
    
    for pattern in patterns:
        print(f"Resource: {pattern.resource_id}")
        print(f"  Stability Score: {pattern.stability_score:.3f}")
        print(f"  Predictability Score: {pattern.predictability_score:.3f}")
        print(f"  Average Usage: {pattern.average_usage:.2f} hours")
        print(f"  Trend: {pattern.trend_direction}")
        print(f"  Seasonal Patterns: {pattern.seasonal_patterns}")
        print()

@dataclass
class RIRecommendation:
    """Reserved instance recommendation"""
    resource_id: str
    instance_type: str
    region: str
    recommended_term: CommitmentTerm
    recommended_payment: PaymentOption
    quantity: int
    annual_savings: float
    upfront_cost: float
    monthly_cost: float
    roi_percentage: float
    payback_months: float
    risk_score: float
    confidence_score: float
    rationale: str


@dataclass
class OptimizationCriteria:
    """Criteria for RI optimization"""
    cost_weight: float = 0.4
    flexibility_weight: float = 0.3
    risk_weight: float = 0.3
    min_stability_score: float = 0.7
    min_predictability_score: float = 0.6
    max_payback_months: int = 12


class RIRecommendationEngine:
    """
    Intelligent RI recommendation engine with multi-dimensional optimization.
    Considers cost savings, flexibility, and risk factors.
    """
    
    def __init__(self, provider: CloudProvider = CloudProvider.AWS, pricing_service=None):
        self.provider = provider
        self.logger = logging.getLogger(__name__ + ".RIRecommendationEngine")
        
        # Use real AWS Pricing API service
        if pricing_service is None:
            from .aws_pricing_service import get_pricing_service
            self.pricing_service = get_pricing_service()
        else:
            self.pricing_service = pricing_service
        
        # Fallback pricing data for offline/testing
        self.pricing_data = self._initialize_pricing_data()
        
    def generate_ri_recommendations(self, usage_patterns: List[UsagePattern],
                                  criteria: OptimizationCriteria = None) -> List[RIRecommendation]:
        """
        Generate RI recommendations based on usage patterns and optimization criteria.
        
        Args:
            usage_patterns: Analyzed usage patterns
            criteria: Optimization criteria (uses defaults if None)
            
        Returns:
            List of RI recommendations ranked by optimization score
        """
        if criteria is None:
            criteria = OptimizationCriteria()
            
        self.logger.info(f"Generating RI recommendations for {len(usage_patterns)} usage patterns")
        
        recommendations = []
        
        for pattern in usage_patterns:
            # Filter patterns that meet minimum criteria
            if not self._meets_ri_criteria(pattern, criteria):
                self.logger.debug(f"Pattern {pattern.resource_id} doesn't meet RI criteria")
                continue
            
            # Generate recommendations for different commitment options
            pattern_recommendations = self._generate_pattern_recommendations(pattern, criteria)
            recommendations.extend(pattern_recommendations)
        
        # Rank recommendations by optimization score
        ranked_recommendations = self._rank_recommendations(recommendations, criteria)
        
        self.logger.info(f"Generated {len(ranked_recommendations)} RI recommendations")
        return ranked_recommendations
    
    def calculate_optimization_score(self, recommendation: RIRecommendation,
                                   criteria: OptimizationCriteria) -> float:
        """
        Calculate multi-dimensional optimization score.
        
        Args:
            recommendation: RI recommendation to score
            criteria: Optimization criteria with weights
            
        Returns:
            Optimization score (0.0 to 1.0, higher is better)
        """
        # Cost score (based on ROI percentage)
        cost_score = min(1.0, recommendation.roi_percentage / 50.0)  # Normalize to 50% max ROI
        
        # Flexibility score (inverse of commitment term and upfront payment)
        flexibility_score = self._calculate_flexibility_score(recommendation)
        
        # Risk score (inverse of risk_score, higher confidence = lower risk)
        risk_score = max(0.0, 1.0 - recommendation.risk_score)
        
        # Weighted combination
        optimization_score = (
            criteria.cost_weight * cost_score +
            criteria.flexibility_weight * flexibility_score +
            criteria.risk_weight * risk_score
        )
        
        return min(1.0, optimization_score)
    
    def compare_commitment_options(self, usage_pattern: UsagePattern) -> Dict[str, RIRecommendation]:
        """
        Compare different commitment options for a usage pattern.
        
        Args:
            usage_pattern: Usage pattern to analyze
            
        Returns:
            Dictionary of recommendations by commitment option
        """
        recommendations = {}
        
        # Generate recommendations for all commitment combinations
        for term in CommitmentTerm:
            for payment in PaymentOption:
                rec = self._calculate_ri_recommendation(usage_pattern, term, payment)
                if rec:
                    key = f"{term.value}_{payment.value}"
                    recommendations[key] = rec
        
        return recommendations
    
    def _meets_ri_criteria(self, pattern: UsagePattern, criteria: OptimizationCriteria) -> bool:
        """Check if usage pattern meets RI recommendation criteria"""
        return (
            pattern.stability_score >= criteria.min_stability_score and
            pattern.predictability_score >= criteria.min_predictability_score and
            pattern.average_usage >= 4.0  # Minimum 4 hours average usage
        )
    
    def _generate_pattern_recommendations(self, pattern: UsagePattern,
                                        criteria: OptimizationCriteria) -> List[RIRecommendation]:
        """Generate all possible RI recommendations for a usage pattern"""
        recommendations = []
        
        for term in CommitmentTerm:
            for payment in PaymentOption:
                rec = self._calculate_ri_recommendation(pattern, term, payment)
                if rec and rec.payback_months <= criteria.max_payback_months:
                    recommendations.append(rec)
        
        return recommendations
    
    def _calculate_ri_recommendation(self, pattern: UsagePattern,
                                   term: CommitmentTerm,
                                   payment: PaymentOption) -> Optional[RIRecommendation]:
        """Calculate RI recommendation for specific term and payment option"""
        try:
            # Get pricing for this configuration
            pricing = self._get_ri_pricing(pattern.instance_type, pattern.region, term, payment)
            if not pricing:
                return None
            
            # Calculate recommended quantity (conservative approach)
            recommended_hours = min(pattern.average_usage, pattern.minimum_usage * 1.2)
            quantity = max(1, int(recommended_hours / 24))  # Convert to instance count
            
            # Calculate costs
            on_demand_cost = self._calculate_on_demand_cost(pattern, quantity)
            ri_cost = self._calculate_ri_cost(pricing, quantity, term)
            
            annual_savings = on_demand_cost - ri_cost['total_annual']
            
            if annual_savings <= 0:
                return None
            
            # Calculate financial metrics
            roi_percentage = (annual_savings / ri_cost['total_annual']) * 100
            payback_months = ri_cost['upfront'] / (annual_savings / 12) if annual_savings > 0 else float('inf')
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(pattern, term, payment)
            
            # Calculate confidence score
            confidence_score = (pattern.stability_score + pattern.predictability_score) / 2
            
            return RIRecommendation(
                resource_id=pattern.resource_id,
                instance_type=pattern.instance_type,
                region=pattern.region,
                recommended_term=term,
                recommended_payment=payment,
                quantity=quantity,
                annual_savings=annual_savings,
                upfront_cost=ri_cost['upfront'],
                monthly_cost=ri_cost['monthly'],
                roi_percentage=roi_percentage,
                payback_months=payback_months,
                risk_score=risk_score,
                confidence_score=confidence_score,
                rationale=self._generate_rationale(pattern, term, payment, annual_savings)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating RI recommendation: {e}")
            return None
    
    def _rank_recommendations(self, recommendations: List[RIRecommendation],
                            criteria: OptimizationCriteria) -> List[RIRecommendation]:
        """Rank recommendations by optimization score"""
        scored_recommendations = []
        
        for rec in recommendations:
            score = self.calculate_optimization_score(rec, criteria)
            scored_recommendations.append((score, rec))
        
        # Sort by score (descending)
        scored_recommendations.sort(key=lambda x: x[0], reverse=True)
        
        return [rec for score, rec in scored_recommendations]
    
    def _calculate_flexibility_score(self, recommendation: RIRecommendation) -> float:
        """Calculate flexibility score based on commitment terms"""
        # Shorter terms and less upfront payment = more flexibility
        term_score = 1.0 if recommendation.recommended_term == CommitmentTerm.ONE_YEAR else 0.5
        
        payment_scores = {
            PaymentOption.NO_UPFRONT: 1.0,
            PaymentOption.PARTIAL_UPFRONT: 0.7,
            PaymentOption.ALL_UPFRONT: 0.3
        }
        payment_score = payment_scores.get(recommendation.recommended_payment, 0.5)
        
        return (term_score + payment_score) / 2
    
    def _calculate_risk_score(self, pattern: UsagePattern,
                            term: CommitmentTerm, payment: PaymentOption) -> float:
        """Calculate risk score for RI recommendation"""
        # Base risk from usage pattern uncertainty
        stability_risk = 1.0 - pattern.stability_score
        predictability_risk = 1.0 - pattern.predictability_score
        
        # Term risk (longer terms = higher risk)
        term_risk = 0.3 if term == CommitmentTerm.ONE_YEAR else 0.7
        
        # Payment risk (more upfront = higher risk)
        payment_risks = {
            PaymentOption.NO_UPFRONT: 0.1,
            PaymentOption.PARTIAL_UPFRONT: 0.4,
            PaymentOption.ALL_UPFRONT: 0.8
        }
        payment_risk = payment_risks.get(payment, 0.5)
        
        # Trend risk
        trend_risk = 0.3 if pattern.trend_direction == "decreasing" else 0.1
        
        # Combined risk score
        total_risk = (stability_risk + predictability_risk + term_risk + payment_risk + trend_risk) / 5
        
        return min(1.0, total_risk)
    
    def _generate_rationale(self, pattern: UsagePattern, term: CommitmentTerm,
                          payment: PaymentOption, savings: float) -> str:
        """Generate human-readable rationale for recommendation"""
        rationale_parts = []
        
        rationale_parts.append(f"Resource shows {pattern.stability_score:.1%} stability")
        rationale_parts.append(f"and {pattern.predictability_score:.1%} predictability")
        rationale_parts.append(f"with {term.value.replace('_', '-')} {payment.value.replace('_', ' ')} commitment")
        rationale_parts.append(f"providing ${savings:,.2f} annual savings")
        
        if pattern.trend_direction == "increasing":
            rationale_parts.append("Usage trend is increasing, supporting RI investment")
        elif pattern.trend_direction == "stable":
            rationale_parts.append("Stable usage pattern ideal for RI commitment")
        
        return ". ".join(rationale_parts) + "."
    
    def _initialize_pricing_data(self) -> Dict[str, Any]:
        """Initialize pricing data (would fetch from cloud provider APIs)"""
        # Sample pricing data for AWS EC2 instances
        return {
            "m5.large": {
                "on_demand_hourly": 0.096,
                "ri_pricing": {
                    CommitmentTerm.ONE_YEAR: {
                        PaymentOption.NO_UPFRONT: {"hourly": 0.069, "upfront": 0},
                        PaymentOption.PARTIAL_UPFRONT: {"hourly": 0.034, "upfront": 312},
                        PaymentOption.ALL_UPFRONT: {"hourly": 0, "upfront": 608}
                    },
                    CommitmentTerm.THREE_YEAR: {
                        PaymentOption.NO_UPFRONT: {"hourly": 0.062, "upfront": 0},
                        PaymentOption.PARTIAL_UPFRONT: {"hourly": 0.030, "upfront": 555},
                        PaymentOption.ALL_UPFRONT: {"hourly": 0, "upfront": 1051}
                    }
                }
            },
            "m5.xlarge": {
                "on_demand_hourly": 0.192,
                "ri_pricing": {
                    CommitmentTerm.ONE_YEAR: {
                        PaymentOption.NO_UPFRONT: {"hourly": 0.138, "upfront": 0},
                        PaymentOption.PARTIAL_UPFRONT: {"hourly": 0.068, "upfront": 624},
                        PaymentOption.ALL_UPFRONT: {"hourly": 0, "upfront": 1216}
                    },
                    CommitmentTerm.THREE_YEAR: {
                        PaymentOption.NO_UPFRONT: {"hourly": 0.124, "upfront": 0},
                        PaymentOption.PARTIAL_UPFRONT: {"hourly": 0.060, "upfront": 1110},
                        PaymentOption.ALL_UPFRONT: {"hourly": 0, "upfront": 2102}
                    }
                }
            },
            "m5.2xlarge": {
                "on_demand_hourly": 0.384,
                "ri_pricing": {
                    CommitmentTerm.ONE_YEAR: {
                        PaymentOption.NO_UPFRONT: {"hourly": 0.276, "upfront": 0},
                        PaymentOption.PARTIAL_UPFRONT: {"hourly": 0.136, "upfront": 1248},
                        PaymentOption.ALL_UPFRONT: {"hourly": 0, "upfront": 2432}
                    },
                    CommitmentTerm.THREE_YEAR: {
                        PaymentOption.NO_UPFRONT: {"hourly": 0.248, "upfront": 0},
                        PaymentOption.PARTIAL_UPFRONT: {"hourly": 0.120, "upfront": 2220},
                        PaymentOption.ALL_UPFRONT: {"hourly": 0, "upfront": 4204}
                    }
                }
            }
        }
    
    def _get_ri_pricing(self, instance_type: str, region: str,
                       term: CommitmentTerm, payment: PaymentOption) -> Optional[Dict[str, float]]:
        """Get RI pricing for specific configuration - now uses real AWS Pricing API"""
        try:
            # Try to fetch real pricing from AWS
            import asyncio
            
            # Convert term and payment to AWS API format
            term_years = 1 if term == CommitmentTerm.ONE_YEAR else 3
            payment_map = {
                PaymentOption.NO_UPFRONT: 'No Upfront',
                PaymentOption.PARTIAL_UPFRONT: 'Partial Upfront',
                PaymentOption.ALL_UPFRONT: 'All Upfront'
            }
            payment_option = payment_map.get(payment, 'No Upfront')
            
            # Fetch real pricing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            pricing_data = loop.run_until_complete(
                self.pricing_service.get_ec2_reserved_pricing(
                    instance_type=instance_type,
                    region=region,
                    term_years=term_years,
                    payment_option=payment_option
                )
            )
            loop.close()
            
            if pricing_data:
                return {
                    'upfront': float(pricing_data['upfront']),
                    'hourly': float(pricing_data['hourly'])
                }
        except Exception as e:
            self.logger.warning(f"Failed to fetch real pricing, using fallback: {e}")
        
        # Fallback to hardcoded pricing if API fails
        if instance_type not in self.pricing_data:
            return None
        
        pricing = self.pricing_data[instance_type]["ri_pricing"]
        if term not in pricing or payment not in pricing[term]:
            return None
        
        return pricing[term][payment]
    
    def _calculate_on_demand_cost(self, pattern: UsagePattern, quantity: int) -> float:
        """Calculate annual on-demand cost"""
        if pattern.instance_type not in self.pricing_data:
            return 0.0
        
        hourly_rate = self.pricing_data[pattern.instance_type]["on_demand_hourly"]
        annual_hours = pattern.average_usage * 365 * quantity
        
        return annual_hours * hourly_rate
    
    def _calculate_ri_cost(self, pricing: Dict[str, float], quantity: int,
                          term: CommitmentTerm) -> Dict[str, float]:
        """Calculate RI costs"""
        upfront = pricing["upfront"] * quantity
        hourly = pricing["hourly"]
        
        # Calculate for full year
        annual_hours = 8760  # Hours in a year
        monthly_recurring = (hourly * annual_hours * quantity) / 12
        total_annual = upfront + (hourly * annual_hours * quantity)
        
        return {
            "upfront": upfront,
            "monthly": monthly_recurring,
            "total_annual": total_annual
        }


if __name__ == "__main__":
    # Example usage of RI recommendation engine
    engine = RIRecommendationEngine()
    
    # Create sample usage pattern
    sample_pattern = UsagePattern(
        resource_id="i-1234567890abcdef0",
        instance_type="m5.large",
        region="us-east-1",
        average_usage=20.0,  # 20 hours per day average
        peak_usage=24.0,
        minimum_usage=16.0,
        stability_score=0.85,
        predictability_score=0.78,
        seasonal_patterns={"daily": 0.2, "weekly": 0.1},
        trend_direction="stable",
        confidence_level=0.82
    )
    
    # Generate recommendations
    recommendations = engine.generate_ri_recommendations([sample_pattern])
    
    print("RI Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
        print(f"\n{i}. {rec.recommended_term.value} {rec.recommended_payment.value}")
        print(f"   Annual Savings: ${rec.annual_savings:,.2f}")
        print(f"   ROI: {rec.roi_percentage:.1f}%")
        print(f"   Payback: {rec.payback_months:.1f} months")
        print(f"   Risk Score: {rec.risk_score:.2f}")
        print(f"   Rationale: {rec.rationale}")


@dataclass
class SavingsAnalysis:
    """Detailed savings analysis for RI recommendations"""
    recommendation_id: str
    total_savings: float
    monthly_savings: float
    percentage_savings: float
    break_even_months: float
    roi_percentage: float
    net_present_value: float
    internal_rate_of_return: float
    payback_period: float
    total_cost_of_ownership: float


@dataclass
class ScenarioAnalysis:
    """Scenario-based analysis for different commitment strategies"""
    scenario_name: str
    assumptions: Dict[str, Any]
    projected_savings: float
    risk_factors: List[str]
    confidence_level: float
    sensitivity_analysis: Dict[str, float]


@dataclass
class FinancialMetrics:
    """Comprehensive financial metrics for RI analysis"""
    initial_investment: float
    annual_cash_flows: List[float]
    discount_rate: float
    terminal_value: float
    npv: float
    irr: float
    payback_period: float
    profitability_index: float


class SavingsCalculator:
    """
    Advanced savings and ROI calculation system with detailed financial analysis,
    payback period calculations, and scenario modeling.
    """
    
    def __init__(self, discount_rate: float = 0.08):
        self.discount_rate = discount_rate  # 8% default discount rate
        self.logger = logging.getLogger(__name__ + ".SavingsCalculator")
    
    def calculate_detailed_savings(self, recommendation: RIRecommendation,
                                 usage_pattern: UsagePattern) -> SavingsAnalysis:
        """
        Calculate comprehensive savings analysis for an RI recommendation.
        
        Args:
            recommendation: RI recommendation to analyze
            usage_pattern: Associated usage pattern
            
        Returns:
            Detailed savings analysis
        """
        self.logger.info(f"Calculating detailed savings for {recommendation.resource_id}")
        
        # Calculate total costs over commitment period
        commitment_years = 1 if recommendation.recommended_term == CommitmentTerm.ONE_YEAR else 3
        
        # On-demand costs
        annual_on_demand = self._calculate_annual_on_demand_cost(usage_pattern, recommendation.quantity)
        total_on_demand = annual_on_demand * commitment_years
        
        # RI costs
        total_ri_cost = self._calculate_total_ri_cost(recommendation, commitment_years)
        
        # Savings calculations
        total_savings = total_on_demand - total_ri_cost
        monthly_savings = total_savings / (commitment_years * 12)
        percentage_savings = (total_savings / total_on_demand) * 100 if total_on_demand > 0 else 0
        
        # Financial metrics
        break_even_months = self._calculate_break_even_period(recommendation)
        roi_percentage = (total_savings / total_ri_cost) * 100 if total_ri_cost > 0 else 0
        npv = self._calculate_npv(recommendation, commitment_years, usage_pattern)
        irr = self._calculate_irr(recommendation, commitment_years, usage_pattern)
        
        return SavingsAnalysis(
            recommendation_id=f"{recommendation.resource_id}_{recommendation.recommended_term.value}",
            total_savings=total_savings,
            monthly_savings=monthly_savings,
            percentage_savings=percentage_savings,
            break_even_months=break_even_months,
            roi_percentage=roi_percentage,
            net_present_value=npv,
            internal_rate_of_return=irr,
            payback_period=recommendation.payback_months,
            total_cost_of_ownership=total_ri_cost
        )
    
    def calculate_break_even_analysis(self, recommendation: RIRecommendation,
                                    usage_pattern: UsagePattern) -> Dict[str, float]:
        """
        Calculate break-even analysis for different scenarios.
        
        Args:
            recommendation: RI recommendation
            usage_pattern: Usage pattern data
            
        Returns:
            Break-even analysis results
        """
        # Base case break-even
        base_break_even = self._calculate_break_even_period(recommendation)
        
        # Sensitivity analysis - what if usage changes?
        usage_scenarios = {
            "usage_decrease_20%": 0.8,
            "usage_decrease_10%": 0.9,
            "base_case": 1.0,
            "usage_increase_10%": 1.1,
            "usage_increase_20%": 1.2
        }
        
        break_even_scenarios = {}
        
        for scenario, multiplier in usage_scenarios.items():
            # Adjust usage pattern
            adjusted_pattern = UsagePattern(
                resource_id=usage_pattern.resource_id,
                instance_type=usage_pattern.instance_type,
                region=usage_pattern.region,
                average_usage=usage_pattern.average_usage * multiplier,
                peak_usage=usage_pattern.peak_usage * multiplier,
                minimum_usage=usage_pattern.minimum_usage * multiplier,
                stability_score=usage_pattern.stability_score,
                predictability_score=usage_pattern.predictability_score,
                seasonal_patterns=usage_pattern.seasonal_patterns,
                trend_direction=usage_pattern.trend_direction,
                confidence_level=usage_pattern.confidence_level
            )
            
            # Calculate break-even for adjusted scenario
            break_even_scenarios[scenario] = self._calculate_break_even_with_pattern(
                recommendation, adjusted_pattern
            )
        
        return break_even_scenarios
    
    def model_commitment_scenarios(self, usage_pattern: UsagePattern,
                                 scenarios: List[str] = None) -> List[ScenarioAnalysis]:
        """
        Model different commitment scenarios and their financial implications.
        
        Args:
            usage_pattern: Base usage pattern
            scenarios: List of scenario names to model
            
        Returns:
            List of scenario analyses
        """
        if scenarios is None:
            scenarios = ["conservative", "aggressive", "balanced"]
        
        scenario_analyses = []
        
        for scenario_name in scenarios:
            analysis = self._model_scenario(usage_pattern, scenario_name)
            scenario_analyses.append(analysis)
        
        return scenario_analyses
    
    def calculate_portfolio_roi(self, recommendations: List[RIRecommendation],
                              usage_patterns: List[UsagePattern]) -> Dict[str, float]:
        """
        Calculate ROI for entire RI portfolio.
        
        Args:
            recommendations: List of RI recommendations
            usage_patterns: Corresponding usage patterns
            
        Returns:
            Portfolio-level financial metrics
        """
        total_investment = sum(rec.upfront_cost for rec in recommendations)
        total_annual_savings = sum(rec.annual_savings for rec in recommendations)
        
        # Portfolio metrics
        portfolio_roi = (total_annual_savings / total_investment) * 100 if total_investment > 0 else 0
        
        # Weighted average payback period
        weighted_payback = sum(
            rec.payback_months * rec.upfront_cost for rec in recommendations
        ) / total_investment if total_investment > 0 else 0
        
        # Risk-adjusted return
        risk_scores = [rec.risk_score for rec in recommendations]
        avg_risk = statistics.mean(risk_scores) if risk_scores else 0
        risk_adjusted_roi = portfolio_roi * (1 - avg_risk)
        
        return {
            "total_investment": total_investment,
            "total_annual_savings": total_annual_savings,
            "portfolio_roi": portfolio_roi,
            "weighted_payback_months": weighted_payback,
            "average_risk_score": avg_risk,
            "risk_adjusted_roi": risk_adjusted_roi,
            "number_of_recommendations": len(recommendations)
        }
    
    def _calculate_annual_on_demand_cost(self, usage_pattern: UsagePattern, quantity: int) -> float:
        """Calculate annual on-demand cost for usage pattern"""
        # This would typically fetch current pricing from cloud provider
        # Using sample pricing for demonstration
        hourly_rates = {
            "m5.large": 0.096,
            "m5.xlarge": 0.192,
            "m5.2xlarge": 0.384
        }
        
        hourly_rate = hourly_rates.get(usage_pattern.instance_type, 0.096)
        annual_hours = usage_pattern.average_usage * 365 * quantity
        
        return annual_hours * hourly_rate
    
    def _calculate_total_ri_cost(self, recommendation: RIRecommendation, years: int) -> float:
        """Calculate total RI cost over commitment period"""
        return recommendation.upfront_cost + (recommendation.monthly_cost * 12 * years)
    
    def _calculate_break_even_period(self, recommendation: RIRecommendation) -> float:
        """Calculate break-even period in months"""
        if recommendation.annual_savings <= 0:
            return float('inf')
        
        monthly_savings = recommendation.annual_savings / 12
        return recommendation.upfront_cost / monthly_savings if monthly_savings > 0 else float('inf')
    
    def _calculate_break_even_with_pattern(self, recommendation: RIRecommendation,
                                         usage_pattern: UsagePattern) -> float:
        """Calculate break-even with adjusted usage pattern"""
        # Recalculate savings with new usage pattern
        annual_on_demand = self._calculate_annual_on_demand_cost(usage_pattern, recommendation.quantity)
        annual_ri_cost = recommendation.monthly_cost * 12
        annual_savings = annual_on_demand - annual_ri_cost
        
        if annual_savings <= 0:
            return float('inf')
        
        monthly_savings = annual_savings / 12
        return recommendation.upfront_cost / monthly_savings if monthly_savings > 0 else float('inf')
    
    def _calculate_npv(self, recommendation: RIRecommendation, years: int,
                      usage_pattern: UsagePattern) -> float:
        """Calculate Net Present Value"""
        # Initial investment (negative cash flow)
        cash_flows = [-recommendation.upfront_cost]
        
        # Annual savings (positive cash flows)
        annual_savings = recommendation.annual_savings
        for year in range(years):
            # Apply discount rate
            discounted_savings = annual_savings / ((1 + self.discount_rate) ** (year + 1))
            cash_flows.append(discounted_savings)
        
        return sum(cash_flows)
    
    def _calculate_irr(self, recommendation: RIRecommendation, years: int,
                      usage_pattern: UsagePattern) -> float:
        """Calculate Internal Rate of Return (simplified)"""
        # Simplified IRR calculation
        initial_investment = recommendation.upfront_cost
        annual_savings = recommendation.annual_savings
        
        if initial_investment <= 0 or annual_savings <= 0:
            return 0.0
        
        # Approximate IRR using simple formula for annuity
        # More sophisticated calculation would use iterative methods
        if years > 0:
            irr_approx = (annual_savings / initial_investment) - (1 / years)
            return max(0.0, min(1.0, irr_approx)) * 100  # Convert to percentage
        
        return 0.0
    
    def _model_scenario(self, usage_pattern: UsagePattern, scenario_name: str) -> ScenarioAnalysis:
        """Model a specific commitment scenario"""
        scenarios = {
            "conservative": {
                "description": "Conservative approach with shorter commitments",
                "term_preference": CommitmentTerm.ONE_YEAR,
                "payment_preference": PaymentOption.NO_UPFRONT,
                "usage_multiplier": 0.8,  # Assume 20% lower usage
                "risk_factors": ["Usage may decrease", "Market conditions uncertain"]
            },
            "aggressive": {
                "description": "Aggressive approach maximizing savings",
                "term_preference": CommitmentTerm.THREE_YEAR,
                "payment_preference": PaymentOption.ALL_UPFRONT,
                "usage_multiplier": 1.1,  # Assume 10% higher usage
                "risk_factors": ["High upfront investment", "Long commitment period"]
            },
            "balanced": {
                "description": "Balanced approach with moderate risk",
                "term_preference": CommitmentTerm.ONE_YEAR,
                "payment_preference": PaymentOption.PARTIAL_UPFRONT,
                "usage_multiplier": 1.0,  # Current usage level
                "risk_factors": ["Moderate upfront cost", "Reasonable flexibility"]
            }
        }
        
        scenario_config = scenarios.get(scenario_name, scenarios["balanced"])
        
        # Adjust usage pattern for scenario
        adjusted_usage = usage_pattern.average_usage * scenario_config["usage_multiplier"]
        
        # Calculate projected savings (simplified)
        base_annual_cost = self._calculate_annual_on_demand_cost(usage_pattern, 1)
        scenario_savings = base_annual_cost * 0.3  # Assume 30% savings
        
        # Sensitivity analysis
        sensitivity = {
            "usage_change_10%": scenario_savings * 0.1,
            "pricing_change_5%": scenario_savings * 0.05,
            "market_volatility": scenario_savings * 0.15
        }
        
        return ScenarioAnalysis(
            scenario_name=scenario_name,
            assumptions=scenario_config,
            projected_savings=scenario_savings,
            risk_factors=scenario_config["risk_factors"],
            confidence_level=0.75,  # Default confidence
            sensitivity_analysis=sensitivity
        )


if __name__ == "__main__":
    # Example usage of savings calculator
    calculator = SavingsCalculator()
    
    # Sample recommendation and usage pattern
    sample_recommendation = RIRecommendation(
        resource_id="i-1234567890abcdef0",
        instance_type="m5.large",
        region="us-east-1",
        recommended_term=CommitmentTerm.ONE_YEAR,
        recommended_payment=PaymentOption.PARTIAL_UPFRONT,
        quantity=1,
        annual_savings=500.0,
        upfront_cost=312.0,
        monthly_cost=30.0,
        roi_percentage=25.0,
        payback_months=7.5,
        risk_score=0.3,
        confidence_score=0.8,
        rationale="Sample recommendation"
    )
    
    sample_pattern = UsagePattern(
        resource_id="i-1234567890abcdef0",
        instance_type="m5.large",
        region="us-east-1",
        average_usage=20.0,
        peak_usage=24.0,
        minimum_usage=16.0,
        stability_score=0.85,
        predictability_score=0.78,
        seasonal_patterns={"daily": 0.2},
        trend_direction="stable",
        confidence_level=0.82
    )
    
    # Calculate detailed savings
    savings_analysis = calculator.calculate_detailed_savings(sample_recommendation, sample_pattern)
    
    print("Detailed Savings Analysis:")
    print(f"Total Savings: ${savings_analysis.total_savings:,.2f}")
    print(f"Monthly Savings: ${savings_analysis.monthly_savings:,.2f}")
    print(f"Percentage Savings: {savings_analysis.percentage_savings:.1f}%")
    print(f"Break-even: {savings_analysis.break_even_months:.1f} months")
    print(f"ROI: {savings_analysis.roi_percentage:.1f}%")
    print(f"NPV: ${savings_analysis.net_present_value:,.2f}")
    
    # Break-even analysis
    break_even_scenarios = calculator.calculate_break_even_analysis(sample_recommendation, sample_pattern)
    print(f"\nBreak-even Scenarios:")
    for scenario, months in break_even_scenarios.items():
        print(f"  {scenario}: {months:.1f} months")
    
    # Scenario modeling
    scenarios = calculator.model_commitment_scenarios(sample_pattern)
    print(f"\nCommitment Scenarios:")
    for scenario in scenarios:
        print(f"  {scenario.scenario_name}: ${scenario.projected_savings:,.2f} projected savings")
        print(f"    Risk factors: {', '.join(scenario.risk_factors)}")


@dataclass
class ReservedInstance:
    """Reserved instance information"""
    ri_id: str
    instance_type: str
    region: str
    availability_zone: str
    term: CommitmentTerm
    payment_option: PaymentOption
    quantity: int
    start_date: datetime
    end_date: datetime
    hourly_rate: float
    upfront_cost: float
    state: str  # active, retired, etc.


@dataclass
class UtilizationReport:
    """RI utilization report"""
    ri_id: str
    reporting_period: Tuple[datetime, datetime]
    total_hours_available: float
    total_hours_used: float
    utilization_percentage: float
    coverage_percentage: float
    unused_hours: float
    cost_savings_realized: float
    cost_savings_potential: float
    efficiency_score: float


@dataclass
class UtilizationAlert:
    """Alert for RI utilization issues"""
    alert_id: str
    ri_id: str
    alert_type: str  # underutilization, overutilization, expiring
    severity: str  # low, medium, high, critical
    message: str
    recommended_actions: List[str]
    created_at: datetime


@dataclass
class CoverageAnalysis:
    """Analysis of RI coverage gaps"""
    resource_type: str
    region: str
    total_usage: float
    ri_coverage: float
    coverage_percentage: float
    gap_hours: float
    gap_cost: float
    recommended_ri_quantity: int


class UtilizationTracker:
    """
    Real-time RI utilization tracking system with alerts and optimization notifications.
    Monitors RI usage, identifies coverage gaps, and provides optimization recommendations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".UtilizationTracker")
        self.utilization_threshold_low = 0.7  # 70% utilization threshold
        self.utilization_threshold_high = 0.95  # 95% utilization threshold
        self.alert_history: List[UtilizationAlert] = []
    
    def track_ri_utilization(self, reserved_instances: List[ReservedInstance],
                           usage_data: List[UsageDataPoint],
                           reporting_period: Tuple[datetime, datetime]) -> List[UtilizationReport]:
        """
        Track utilization for all reserved instances.
        
        Args:
            reserved_instances: List of reserved instances to track
            usage_data: Actual usage data for the period
            reporting_period: Start and end dates for reporting
            
        Returns:
            List of utilization reports for each RI
        """
        self.logger.info(f"Tracking utilization for {len(reserved_instances)} RIs")
        
        reports = []
        
        for ri in reserved_instances:
            # Filter usage data for this RI
            ri_usage = self._filter_usage_for_ri(usage_data, ri, reporting_period)
            
            # Calculate utilization metrics
            report = self._calculate_ri_utilization(ri, ri_usage, reporting_period)
            reports.append(report)
            
            # Check for alerts
            alerts = self._check_utilization_alerts(ri, report)
            self.alert_history.extend(alerts)
        
        self.logger.info(f"Generated {len(reports)} utilization reports")
        return reports
    
    def analyze_coverage_gaps(self, usage_data: List[UsageDataPoint],
                            reserved_instances: List[ReservedInstance],
                            analysis_period: Tuple[datetime, datetime]) -> List[CoverageAnalysis]:
        """
        Analyze RI coverage gaps and identify opportunities for additional RIs.
        
        Args:
            usage_data: Actual usage data
            reserved_instances: Current reserved instances
            analysis_period: Period for analysis
            
        Returns:
            List of coverage gap analyses
        """
        self.logger.info("Analyzing RI coverage gaps")
        
        # Group usage by resource type and region
        usage_by_type_region = self._group_usage_by_type_region(usage_data, analysis_period)
        
        # Group RIs by type and region
        ri_by_type_region = self._group_ri_by_type_region(reserved_instances)
        
        coverage_analyses = []
        
        for (instance_type, region), usage_hours in usage_by_type_region.items():
            # Calculate RI coverage for this type/region
            ri_coverage = ri_by_type_region.get((instance_type, region), 0)
            
            # Calculate coverage metrics
            coverage_percentage = (ri_coverage / usage_hours) * 100 if usage_hours > 0 else 0
            gap_hours = max(0, usage_hours - ri_coverage)
            
            # Estimate gap cost (using sample pricing)
            hourly_rate = 0.096  # Sample rate for m5.large
            gap_cost = gap_hours * hourly_rate
            
            # Recommend additional RIs if significant gap
            recommended_quantity = int(gap_hours / (24 * 30)) if gap_hours > 0 else 0  # Monthly equivalent
            
            if gap_hours > 0:  # Only include if there's a gap
                coverage_analyses.append(CoverageAnalysis(
                    resource_type=instance_type,
                    region=region,
                    total_usage=usage_hours,
                    ri_coverage=ri_coverage,
                    coverage_percentage=coverage_percentage,
                    gap_hours=gap_hours,
                    gap_cost=gap_cost,
                    recommended_ri_quantity=recommended_quantity
                ))
        
        self.logger.info(f"Identified {len(coverage_analyses)} coverage gaps")
        return coverage_analyses
    
    def generate_utilization_alerts(self, utilization_reports: List[UtilizationReport]) -> List[UtilizationAlert]:
        """
        Generate alerts based on utilization reports.
        
        Args:
            utilization_reports: List of utilization reports
            
        Returns:
            List of utilization alerts
        """
        alerts = []
        
        for report in utilization_reports:
            # Check for underutilization
            if report.utilization_percentage < self.utilization_threshold_low * 100:
                severity = "high" if report.utilization_percentage < 50 else "medium"
                
                alert = UtilizationAlert(
                    alert_id=f"underutil_{report.ri_id}_{datetime.now().strftime('%Y%m%d')}",
                    ri_id=report.ri_id,
                    alert_type="underutilization",
                    severity=severity,
                    message=f"RI {report.ri_id} is only {report.utilization_percentage:.1f}% utilized",
                    recommended_actions=[
                        "Consider modifying RI to smaller instance type",
                        "Review workload placement strategy",
                        "Consider selling unused capacity"
                    ],
                    created_at=datetime.now()
                )
                alerts.append(alert)
            
            # Check for overutilization (indicates need for more RIs)
            elif report.utilization_percentage > self.utilization_threshold_high * 100:
                alert = UtilizationAlert(
                    alert_id=f"overutil_{report.ri_id}_{datetime.now().strftime('%Y%m%d')}",
                    ri_id=report.ri_id,
                    alert_type="overutilization",
                    severity="medium",
                    message=f"RI {report.ri_id} is {report.utilization_percentage:.1f}% utilized - consider additional capacity",
                    recommended_actions=[
                        "Purchase additional RIs for this instance type",
                        "Consider upgrading to larger instance type",
                        "Review capacity planning"
                    ],
                    created_at=datetime.now()
                )
                alerts.append(alert)
        
        return alerts
    
    def monitor_ri_expiration(self, reserved_instances: List[ReservedInstance],
                            warning_days: int = 90) -> List[UtilizationAlert]:
        """
        Monitor RI expiration dates and generate alerts.
        
        Args:
            reserved_instances: List of reserved instances
            warning_days: Days before expiration to alert
            
        Returns:
            List of expiration alerts
        """
        alerts = []
        current_date = datetime.now()
        warning_date = current_date + timedelta(days=warning_days)
        
        for ri in reserved_instances:
            if ri.end_date <= warning_date and ri.state == "active":
                days_until_expiry = (ri.end_date - current_date).days
                
                severity = "critical" if days_until_expiry <= 30 else "high"
                
                alert = UtilizationAlert(
                    alert_id=f"expiring_{ri.ri_id}_{current_date.strftime('%Y%m%d')}",
                    ri_id=ri.ri_id,
                    alert_type="expiring",
                    severity=severity,
                    message=f"RI {ri.ri_id} expires in {days_until_expiry} days",
                    recommended_actions=[
                        "Review renewal options",
                        "Analyze current utilization",
                        "Consider alternative commitment options",
                        "Plan for capacity replacement"
                    ],
                    created_at=current_date
                )
                alerts.append(alert)
        
        return alerts
    
    def calculate_efficiency_metrics(self, utilization_reports: List[UtilizationReport]) -> Dict[str, float]:
        """
        Calculate overall RI portfolio efficiency metrics.
        
        Args:
            utilization_reports: List of utilization reports
            
        Returns:
            Dictionary of efficiency metrics
        """
        if not utilization_reports:
            return {}
        
        # Overall utilization
        total_available = sum(report.total_hours_available for report in utilization_reports)
        total_used = sum(report.total_hours_used for report in utilization_reports)
        overall_utilization = (total_used / total_available) * 100 if total_available > 0 else 0
        
        # Coverage metrics
        coverage_percentages = [report.coverage_percentage for report in utilization_reports]
        average_coverage = statistics.mean(coverage_percentages) if coverage_percentages else 0
        
        # Savings metrics
        total_savings_realized = sum(report.cost_savings_realized for report in utilization_reports)
        total_savings_potential = sum(report.cost_savings_potential for report in utilization_reports)
        savings_efficiency = (total_savings_realized / total_savings_potential) * 100 if total_savings_potential > 0 else 0
        
        # Efficiency scores
        efficiency_scores = [report.efficiency_score for report in utilization_reports]
        average_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0
        
        return {
            "overall_utilization_percentage": overall_utilization,
            "average_coverage_percentage": average_coverage,
            "total_savings_realized": total_savings_realized,
            "total_savings_potential": total_savings_potential,
            "savings_efficiency_percentage": savings_efficiency,
            "average_efficiency_score": average_efficiency,
            "number_of_ris": len(utilization_reports),
            "underutilized_ris": len([r for r in utilization_reports if r.utilization_percentage < 70]),
            "well_utilized_ris": len([r for r in utilization_reports if 70 <= r.utilization_percentage <= 95])
        }
    
    def _filter_usage_for_ri(self, usage_data: List[UsageDataPoint],
                           ri: ReservedInstance,
                           period: Tuple[datetime, datetime]) -> List[UsageDataPoint]:
        """Filter usage data relevant to a specific RI"""
        start_date, end_date = period
        
        return [
            usage for usage in usage_data
            if (usage.instance_type == ri.instance_type and
                usage.region == ri.region and
                start_date <= usage.timestamp <= end_date)
        ]
    
    def _calculate_ri_utilization(self, ri: ReservedInstance,
                                usage_data: List[UsageDataPoint],
                                period: Tuple[datetime, datetime]) -> UtilizationReport:
        """Calculate utilization metrics for a single RI"""
        start_date, end_date = period
        period_hours = (end_date - start_date).total_seconds() / 3600
        
        # Total hours available from RI
        total_hours_available = period_hours * ri.quantity
        
        # Total hours actually used
        total_hours_used = sum(usage.usage_hours for usage in usage_data)
        
        # Utilization percentage
        utilization_percentage = (total_hours_used / total_hours_available) * 100 if total_hours_available > 0 else 0
        
        # Coverage percentage (how much of actual usage is covered by RI)
        actual_instance_hours = len(usage_data)  # Simplified - would be more complex in reality
        coverage_percentage = min(100, (total_hours_available / actual_instance_hours) * 100) if actual_instance_hours > 0 else 0
        
        # Unused hours
        unused_hours = max(0, total_hours_available - total_hours_used)
        
        # Cost savings (simplified calculation)
        on_demand_rate = 0.096  # Sample rate
        ri_rate = ri.hourly_rate
        cost_savings_realized = total_hours_used * (on_demand_rate - ri_rate)
        cost_savings_potential = total_hours_available * (on_demand_rate - ri_rate)
        
        # Efficiency score (combination of utilization and cost effectiveness)
        efficiency_score = (utilization_percentage / 100) * (cost_savings_realized / cost_savings_potential) if cost_savings_potential > 0 else 0
        
        return UtilizationReport(
            ri_id=ri.ri_id,
            reporting_period=period,
            total_hours_available=total_hours_available,
            total_hours_used=total_hours_used,
            utilization_percentage=utilization_percentage,
            coverage_percentage=coverage_percentage,
            unused_hours=unused_hours,
            cost_savings_realized=cost_savings_realized,
            cost_savings_potential=cost_savings_potential,
            efficiency_score=efficiency_score
        )
    
    def _check_utilization_alerts(self, ri: ReservedInstance,
                                report: UtilizationReport) -> List[UtilizationAlert]:
        """Check if RI utilization triggers any alerts"""
        alerts = []
        
        # Low utilization alert
        if report.utilization_percentage < self.utilization_threshold_low * 100:
            severity = "critical" if report.utilization_percentage < 30 else "high"
            
            alert = UtilizationAlert(
                alert_id=f"low_util_{ri.ri_id}_{datetime.now().strftime('%Y%m%d%H%M')}",
                ri_id=ri.ri_id,
                alert_type="underutilization",
                severity=severity,
                message=f"Low utilization: {report.utilization_percentage:.1f}%",
                recommended_actions=[
                    "Review workload placement",
                    "Consider RI modification",
                    "Optimize resource allocation"
                ],
                created_at=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    def _group_usage_by_type_region(self, usage_data: List[UsageDataPoint],
                                  period: Tuple[datetime, datetime]) -> Dict[Tuple[str, str], float]:
        """Group usage data by instance type and region"""
        start_date, end_date = period
        
        usage_groups = defaultdict(float)
        
        for usage in usage_data:
            if start_date <= usage.timestamp <= end_date:
                key = (usage.instance_type, usage.region)
                usage_groups[key] += usage.usage_hours
        
        return dict(usage_groups)
    
    def _group_ri_by_type_region(self, reserved_instances: List[ReservedInstance]) -> Dict[Tuple[str, str], float]:
        """Group RI capacity by instance type and region"""
        ri_groups = defaultdict(float)
        
        for ri in reserved_instances:
            if ri.state == "active":
                key = (ri.instance_type, ri.region)
                # Calculate monthly hours for this RI
                monthly_hours = 24 * 30 * ri.quantity  # Simplified monthly calculation
                ri_groups[key] += monthly_hours
        
        return dict(ri_groups)


if __name__ == "__main__":
    # Example usage of utilization tracker
    tracker = UtilizationTracker()
    
    # Sample reserved instance
    sample_ri = ReservedInstance(
        ri_id="ri-1234567890abcdef0",
        instance_type="m5.large",
        region="us-east-1",
        availability_zone="us-east-1a",
        term=CommitmentTerm.ONE_YEAR,
        payment_option=PaymentOption.PARTIAL_UPFRONT,
        quantity=2,
        start_date=datetime.now() - timedelta(days=180),
        end_date=datetime.now() + timedelta(days=185),
        hourly_rate=0.069,
        upfront_cost=624.0,
        state="active"
    )
    
    # Generate sample usage data
    sample_usage = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(30 * 24):  # 30 days of hourly data
        timestamp = base_time + timedelta(hours=i)
        
        # Simulate varying utilization (70% average)
        base_usage = 1.4  # 1.4 instances on average (70% of 2 RIs)
        variation = 0.3 * np.sin(2 * np.pi * i / 24)  # Daily pattern
        noise = np.random.normal(0, 0.1)
        
        usage_hours = max(0, base_usage + variation + noise)
        
        sample_usage.append(UsageDataPoint(
            timestamp=timestamp,
            resource_id=f"i-{i:010d}",
            instance_type="m5.large",
            region="us-east-1",
            usage_hours=usage_hours,
            cost=usage_hours * 0.096
        ))
    
    # Track utilization
    period = (base_time, datetime.now())
    reports = tracker.track_ri_utilization([sample_ri], sample_usage, period)
    
    for report in reports:
        print(f"RI Utilization Report for {report.ri_id}:")
        print(f"  Utilization: {report.utilization_percentage:.1f}%")
        print(f"  Coverage: {report.coverage_percentage:.1f}%")
        print(f"  Hours Available: {report.total_hours_available:.0f}")
        print(f"  Hours Used: {report.total_hours_used:.0f}")
        print(f"  Unused Hours: {report.unused_hours:.0f}")
        print(f"  Cost Savings Realized: ${report.cost_savings_realized:.2f}")
        print(f"  Efficiency Score: {report.efficiency_score:.3f}")
    
    # Generate alerts
    alerts = tracker.generate_utilization_alerts(reports)
    print(f"\nGenerated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  {alert.alert_type.upper()} ({alert.severity}): {alert.message}")
    
    # Calculate efficiency metrics
    metrics = tracker.calculate_efficiency_metrics(reports)
    print(f"\nPortfolio Efficiency Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")


@dataclass
class PortfolioOverview:
    """Overview of RI portfolio performance"""
    total_ris: int
    total_investment: float
    total_annual_savings: float
    portfolio_roi: float
    average_utilization: float
    total_coverage: float
    expiring_soon: int
    underutilized: int
    optimization_opportunities: int
    risk_score: float


@dataclass
class ModificationRecommendation:
    """Recommendation for RI modification or exchange"""
    ri_id: str
    modification_type: str  # exchange, modify, terminate
    current_config: Dict[str, Any]
    recommended_config: Dict[str, Any]
    expected_savings: float
    implementation_effort: str
    risk_level: str
    rationale: str


@dataclass
class CapacityPlan:
    """Capacity planning for future RI needs"""
    planning_horizon_months: int
    projected_growth: float
    recommended_purchases: List[Dict[str, Any]]
    budget_requirements: float
    risk_factors: List[str]
    confidence_level: float


class RIPortfolioManager:
    """
    Comprehensive RI portfolio management system with performance analytics,
    modification recommendations, and capacity planning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".RIPortfolioManager")
        self.modification_threshold = 0.6  # 60% utilization threshold for modifications
        self.expiry_warning_days = 90
    
    def generate_portfolio_overview(self, reserved_instances: List[ReservedInstance],
                                  utilization_reports: List[UtilizationReport],
                                  savings_analyses: List[SavingsAnalysis]) -> PortfolioOverview:
        """
        Generate comprehensive portfolio overview with key metrics.
        
        Args:
            reserved_instances: List of all reserved instances
            utilization_reports: Recent utilization reports
            savings_analyses: Savings analysis for each RI
            
        Returns:
            Portfolio overview with key performance metrics
        """
        self.logger.info("Generating portfolio overview")
        
        # Basic counts and totals
        total_ris = len(reserved_instances)
        total_investment = sum(ri.upfront_cost for ri in reserved_instances)
        
        # Savings metrics
        total_annual_savings = sum(analysis.total_savings for analysis in savings_analyses)
        portfolio_roi = (total_annual_savings / total_investment) * 100 if total_investment > 0 else 0
        
        # Utilization metrics
        utilization_percentages = [report.utilization_percentage for report in utilization_reports]
        average_utilization = statistics.mean(utilization_percentages) if utilization_percentages else 0
        
        coverage_percentages = [report.coverage_percentage for report in utilization_reports]
        total_coverage = statistics.mean(coverage_percentages) if coverage_percentages else 0
        
        # Status counts
        current_date = datetime.now()
        expiry_threshold = current_date + timedelta(days=self.expiry_warning_days)
        
        expiring_soon = len([ri for ri in reserved_instances 
                           if ri.end_date <= expiry_threshold and ri.state == "active"])
        
        underutilized = len([report for report in utilization_reports 
                           if report.utilization_percentage < self.modification_threshold * 100])
        
        # Optimization opportunities (simplified)
        optimization_opportunities = underutilized + expiring_soon
        
        # Risk assessment
        risk_factors = []
        if underutilized > total_ris * 0.3:  # More than 30% underutilized
            risk_factors.append("High underutilization")
        if expiring_soon > 0:
            risk_factors.append("Expiring RIs")
        if average_utilization < 70:
            risk_factors.append("Low average utilization")
        
        risk_score = len(risk_factors) / 5.0  # Normalize to 0-1 scale
        
        return PortfolioOverview(
            total_ris=total_ris,
            total_investment=total_investment,
            total_annual_savings=total_annual_savings,
            portfolio_roi=portfolio_roi,
            average_utilization=average_utilization,
            total_coverage=total_coverage,
            expiring_soon=expiring_soon,
            underutilized=underutilized,
            optimization_opportunities=optimization_opportunities,
            risk_score=risk_score
        )
    
    def generate_modification_recommendations(self, reserved_instances: List[ReservedInstance],
                                           utilization_reports: List[UtilizationReport],
                                           usage_patterns: List[UsagePattern]) -> List[ModificationRecommendation]:
        """
        Generate recommendations for RI modifications and exchanges.
        
        Args:
            reserved_instances: Current reserved instances
            utilization_reports: Utilization data
            usage_patterns: Usage pattern analysis
            
        Returns:
            List of modification recommendations
        """
        self.logger.info("Generating modification recommendations")
        
        recommendations = []
        
        # Create lookup dictionaries
        utilization_by_ri = {report.ri_id: report for report in utilization_reports}
        
        for ri in reserved_instances:
            if ri.state != "active":
                continue
                
            utilization_report = utilization_by_ri.get(ri.ri_id)
            if not utilization_report:
                continue
            
            # Check for modification opportunities
            modification_recs = self._analyze_ri_for_modifications(ri, utilization_report, usage_patterns)
            recommendations.extend(modification_recs)
        
        # Sort by expected savings (descending)
        recommendations.sort(key=lambda x: x.expected_savings, reverse=True)
        
        self.logger.info(f"Generated {len(recommendations)} modification recommendations")
        return recommendations
    
    def create_capacity_plan(self, usage_patterns: List[UsagePattern],
                           reserved_instances: List[ReservedInstance],
                           planning_horizon_months: int = 12) -> CapacityPlan:
        """
        Create capacity planning recommendations for future RI needs.
        
        Args:
            usage_patterns: Historical usage patterns
            reserved_instances: Current RI portfolio
            planning_horizon_months: Planning horizon in months
            
        Returns:
            Capacity plan with recommendations
        """
        self.logger.info(f"Creating {planning_horizon_months}-month capacity plan")
        
        # Analyze growth trends
        growth_trends = self._analyze_growth_trends(usage_patterns)
        projected_growth = statistics.mean(growth_trends) if growth_trends else 0.05  # Default 5% growth
        
        # Current capacity analysis
        current_capacity = self._calculate_current_capacity(reserved_instances)
        
        # Project future needs
        future_needs = self._project_future_capacity_needs(usage_patterns, projected_growth, planning_horizon_months)
        
        # Generate purchase recommendations
        recommended_purchases = self._generate_purchase_recommendations(current_capacity, future_needs)
        
        # Calculate budget requirements
        budget_requirements = sum(purchase.get("estimated_cost", 0) for purchase in recommended_purchases)
        
        # Identify risk factors
        risk_factors = self._identify_capacity_risks(usage_patterns, projected_growth)
        
        # Calculate confidence level
        confidence_level = self._calculate_planning_confidence(usage_patterns, growth_trends)
        
        return CapacityPlan(
            planning_horizon_months=planning_horizon_months,
            projected_growth=projected_growth,
            recommended_purchases=recommended_purchases,
            budget_requirements=budget_requirements,
            risk_factors=risk_factors,
            confidence_level=confidence_level
        )
    
    def analyze_portfolio_performance(self, reserved_instances: List[ReservedInstance],
                                   utilization_reports: List[UtilizationReport],
                                   time_period_months: int = 12) -> Dict[str, Any]:
        """
        Analyze portfolio performance over time.
        
        Args:
            reserved_instances: RI portfolio
            utilization_reports: Historical utilization data
            time_period_months: Analysis period
            
        Returns:
            Performance analysis results
        """
        # Performance trends
        utilization_trend = self._calculate_utilization_trend(utilization_reports)
        
        # Cost efficiency
        cost_efficiency = self._calculate_cost_efficiency(reserved_instances, utilization_reports)
        
        # Coverage analysis
        coverage_analysis = self._analyze_coverage_trends(utilization_reports)
        
        # ROI analysis
        roi_analysis = self._calculate_portfolio_roi_trends(reserved_instances, utilization_reports)
        
        return {
            "utilization_trend": utilization_trend,
            "cost_efficiency": cost_efficiency,
            "coverage_analysis": coverage_analysis,
            "roi_analysis": roi_analysis,
            "analysis_period_months": time_period_months,
            "total_ris_analyzed": len(reserved_instances),
            "performance_score": self._calculate_overall_performance_score(utilization_reports)
        }
    
    def optimize_portfolio_allocation(self, usage_patterns: List[UsagePattern],
                                    budget_constraint: float) -> List[RIRecommendation]:
        """
        Optimize RI portfolio allocation within budget constraints.
        
        Args:
            usage_patterns: Usage patterns to optimize for
            budget_constraint: Maximum budget available
            
        Returns:
            Optimized list of RI recommendations within budget
        """
        self.logger.info(f"Optimizing portfolio allocation with ${budget_constraint:,.2f} budget")
        
        # Generate all possible recommendations
        engine = RIRecommendationEngine()
        all_recommendations = engine.generate_ri_recommendations(usage_patterns)
        
        # Sort by ROI (descending)
        all_recommendations.sort(key=lambda x: x.roi_percentage, reverse=True)
        
        # Select recommendations within budget constraint
        selected_recommendations = []
        total_cost = 0
        
        for rec in all_recommendations:
            if total_cost + rec.upfront_cost <= budget_constraint:
                selected_recommendations.append(rec)
                total_cost += rec.upfront_cost
            else:
                # Check if we can fit a smaller quantity
                remaining_budget = budget_constraint - total_cost
                if rec.upfront_cost > 0:
                    affordable_quantity = int(remaining_budget / (rec.upfront_cost / rec.quantity))
                    if affordable_quantity > 0:
                        # Create modified recommendation with affordable quantity
                        modified_rec = RIRecommendation(
                            resource_id=rec.resource_id,
                            instance_type=rec.instance_type,
                            region=rec.region,
                            recommended_term=rec.recommended_term,
                            recommended_payment=rec.recommended_payment,
                            quantity=affordable_quantity,
                            annual_savings=rec.annual_savings * (affordable_quantity / rec.quantity),
                            upfront_cost=rec.upfront_cost * (affordable_quantity / rec.quantity),
                            monthly_cost=rec.monthly_cost * (affordable_quantity / rec.quantity),
                            roi_percentage=rec.roi_percentage,  # ROI percentage stays the same
                            payback_months=rec.payback_months,
                            risk_score=rec.risk_score,
                            confidence_score=rec.confidence_score,
                            rationale=f"Budget-optimized: {rec.rationale}"
                        )
                        selected_recommendations.append(modified_rec)
                        total_cost += modified_rec.upfront_cost
                        break
        
        self.logger.info(f"Selected {len(selected_recommendations)} recommendations using ${total_cost:,.2f} of budget")
        return selected_recommendations
    
    def _analyze_ri_for_modifications(self, ri: ReservedInstance,
                                    utilization_report: UtilizationReport,
                                    usage_patterns: List[UsagePattern]) -> List[ModificationRecommendation]:
        """Analyze a single RI for modification opportunities"""
        recommendations = []
        
        # Low utilization - consider downsizing or exchange
        if utilization_report.utilization_percentage < self.modification_threshold * 100:
            # Recommend exchange to smaller instance type
            current_config = {
                "instance_type": ri.instance_type,
                "quantity": ri.quantity,
                "term": ri.term.value,
                "payment": ri.payment_option.value
            }
            
            # Suggest smaller instance type (simplified logic)
            smaller_types = {
                "m5.2xlarge": "m5.xlarge",
                "m5.xlarge": "m5.large",
                "m5.large": "m5.medium"
            }
            
            recommended_type = smaller_types.get(ri.instance_type)
            if recommended_type:
                recommended_config = current_config.copy()
                recommended_config["instance_type"] = recommended_type
                
                # Estimate savings (simplified)
                expected_savings = utilization_report.unused_hours * 0.05  # $0.05 per unused hour
                
                recommendations.append(ModificationRecommendation(
                    ri_id=ri.ri_id,
                    modification_type="exchange",
                    current_config=current_config,
                    recommended_config=recommended_config,
                    expected_savings=expected_savings,
                    implementation_effort="medium",
                    risk_level="low",
                    rationale=f"Low utilization ({utilization_report.utilization_percentage:.1f}%) suggests downsizing opportunity"
                ))
        
        # Expiring soon - consider renewal or replacement
        days_until_expiry = (ri.end_date - datetime.now()).days
        if days_until_expiry <= self.expiry_warning_days:
            current_config = {
                "instance_type": ri.instance_type,
                "quantity": ri.quantity,
                "term": ri.term.value,
                "payment": ri.payment_option.value
            }
            
            # Recommend renewal with current utilization in mind
            if utilization_report.utilization_percentage >= 70:
                recommended_config = current_config.copy()
                recommended_config["term"] = "3_year"  # Longer term for stable workloads
                
                expected_savings = ri.upfront_cost * 0.15  # Assume 15% additional savings with 3-year term
                
                recommendations.append(ModificationRecommendation(
                    ri_id=ri.ri_id,
                    modification_type="renew",
                    current_config=current_config,
                    recommended_config=recommended_config,
                    expected_savings=expected_savings,
                    implementation_effort="low",
                    risk_level="low",
                    rationale=f"Expiring in {days_until_expiry} days with good utilization ({utilization_report.utilization_percentage:.1f}%)"
                ))
        
        return recommendations
    
    def _analyze_growth_trends(self, usage_patterns: List[UsagePattern]) -> List[float]:
        """Analyze growth trends from usage patterns"""
        growth_rates = []
        
        for pattern in usage_patterns:
            if pattern.trend_direction == "increasing":
                # Estimate growth rate based on confidence level
                growth_rate = 0.1 * pattern.confidence_level  # Up to 10% growth
                growth_rates.append(growth_rate)
            elif pattern.trend_direction == "decreasing":
                growth_rate = -0.05 * pattern.confidence_level  # Up to 5% decline
                growth_rates.append(growth_rate)
            else:  # stable
                growth_rates.append(0.02)  # Assume 2% baseline growth
        
        return growth_rates
    
    def _calculate_current_capacity(self, reserved_instances: List[ReservedInstance]) -> Dict[str, int]:
        """Calculate current RI capacity by instance type"""
        capacity = defaultdict(int)
        
        for ri in reserved_instances:
            if ri.state == "active":
                capacity[ri.instance_type] += ri.quantity
        
        return dict(capacity)
    
    def _project_future_capacity_needs(self, usage_patterns: List[UsagePattern],
                                     growth_rate: float, months: int) -> Dict[str, int]:
        """Project future capacity needs"""
        future_needs = defaultdict(int)
        
        for pattern in usage_patterns:
            # Project usage growth
            current_need = max(1, int(pattern.average_usage / 24))  # Convert to instance count
            future_need = int(current_need * (1 + growth_rate) ** (months / 12))
            future_needs[pattern.instance_type] += future_need
        
        return dict(future_needs)
    
    def _generate_purchase_recommendations(self, current_capacity: Dict[str, int],
                                         future_needs: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate RI purchase recommendations"""
        recommendations = []
        
        for instance_type, future_need in future_needs.items():
            current = current_capacity.get(instance_type, 0)
            additional_needed = max(0, future_need - current)
            
            if additional_needed > 0:
                # Estimate cost (simplified)
                base_costs = {
                    "m5.large": 608,
                    "m5.xlarge": 1216,
                    "m5.2xlarge": 2432
                }
                estimated_cost = base_costs.get(instance_type, 608) * additional_needed
                
                recommendations.append({
                    "instance_type": instance_type,
                    "quantity": additional_needed,
                    "recommended_term": "1_year",
                    "recommended_payment": "partial_upfront",
                    "estimated_cost": estimated_cost,
                    "rationale": f"Projected growth requires {additional_needed} additional {instance_type} instances"
                })
        
        return recommendations
    
    def _identify_capacity_risks(self, usage_patterns: List[UsagePattern], growth_rate: float) -> List[str]:
        """Identify risks in capacity planning"""
        risks = []
        
        if growth_rate > 0.2:  # More than 20% growth
            risks.append("High growth rate may lead to capacity shortfalls")
        
        if growth_rate < -0.1:  # More than 10% decline
            risks.append("Declining usage may result in overcommitment")
        
        # Check pattern stability
        unstable_patterns = [p for p in usage_patterns if p.stability_score < 0.6]
        if len(unstable_patterns) > len(usage_patterns) * 0.3:
            risks.append("High number of unstable usage patterns increases forecast uncertainty")
        
        # Check predictability
        unpredictable_patterns = [p for p in usage_patterns if p.predictability_score < 0.5]
        if len(unpredictable_patterns) > len(usage_patterns) * 0.3:
            risks.append("Low predictability in usage patterns affects planning accuracy")
        
        return risks
    
    def _calculate_planning_confidence(self, usage_patterns: List[UsagePattern],
                                     growth_trends: List[float]) -> float:
        """Calculate confidence level for capacity planning"""
        if not usage_patterns:
            return 0.0
        
        # Average stability and predictability
        avg_stability = statistics.mean([p.stability_score for p in usage_patterns])
        avg_predictability = statistics.mean([p.predictability_score for p in usage_patterns])
        
        # Growth trend consistency
        growth_consistency = 1.0 - (statistics.stdev(growth_trends) if len(growth_trends) > 1 else 0.0)
        
        # Combined confidence
        confidence = (avg_stability + avg_predictability + growth_consistency) / 3
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_utilization_trend(self, utilization_reports: List[UtilizationReport]) -> Dict[str, float]:
        """Calculate utilization trends over time"""
        # Simplified trend calculation
        if len(utilization_reports) < 2:
            return {"trend": 0.0, "direction": "stable"}
        
        utilizations = [report.utilization_percentage for report in utilization_reports]
        
        # Simple linear trend
        x = list(range(len(utilizations)))
        if len(x) > 1:
            # Calculate slope
            n = len(x)
            slope = (n * sum(x[i] * utilizations[i] for i in range(n)) - sum(x) * sum(utilizations)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
            
            direction = "increasing" if slope > 1 else "decreasing" if slope < -1 else "stable"
            
            return {"trend": slope, "direction": direction}
        
        return {"trend": 0.0, "direction": "stable"}
    
    def _calculate_cost_efficiency(self, reserved_instances: List[ReservedInstance],
                                 utilization_reports: List[UtilizationReport]) -> Dict[str, float]:
        """Calculate cost efficiency metrics"""
        total_investment = sum(ri.upfront_cost for ri in reserved_instances)
        
        # Calculate realized savings
        total_realized_savings = sum(report.cost_savings_realized for report in utilization_reports)
        
        efficiency = (total_realized_savings / total_investment) * 100 if total_investment > 0 else 0
        
        return {
            "efficiency_percentage": efficiency,
            "total_investment": total_investment,
            "total_realized_savings": total_realized_savings
        }
    
    def _analyze_coverage_trends(self, utilization_reports: List[UtilizationReport]) -> Dict[str, float]:
        """Analyze coverage trends"""
        if not utilization_reports:
            return {"average_coverage": 0.0, "coverage_trend": 0.0}
        
        coverage_percentages = [report.coverage_percentage for report in utilization_reports]
        average_coverage = statistics.mean(coverage_percentages)
        
        # Simple trend calculation
        if len(coverage_percentages) > 1:
            coverage_trend = coverage_percentages[-1] - coverage_percentages[0]
        else:
            coverage_trend = 0.0
        
        return {
            "average_coverage": average_coverage,
            "coverage_trend": coverage_trend
        }
    
    def _calculate_portfolio_roi_trends(self, reserved_instances: List[ReservedInstance],
                                      utilization_reports: List[UtilizationReport]) -> Dict[str, float]:
        """Calculate portfolio ROI trends"""
        total_investment = sum(ri.upfront_cost for ri in reserved_instances)
        total_savings = sum(report.cost_savings_realized for report in utilization_reports)
        
        current_roi = (total_savings / total_investment) * 100 if total_investment > 0 else 0
        
        return {
            "current_roi": current_roi,
            "total_investment": total_investment,
            "total_savings": total_savings
        }
    
    def _calculate_overall_performance_score(self, utilization_reports: List[UtilizationReport]) -> float:
        """Calculate overall portfolio performance score"""
        if not utilization_reports:
            return 0.0
        
        # Weighted score based on utilization, coverage, and efficiency
        utilization_scores = [min(1.0, report.utilization_percentage / 100) for report in utilization_reports]
        coverage_scores = [min(1.0, report.coverage_percentage / 100) for report in utilization_reports]
        efficiency_scores = [report.efficiency_score for report in utilization_reports]
        
        avg_utilization = statistics.mean(utilization_scores)
        avg_coverage = statistics.mean(coverage_scores)
        avg_efficiency = statistics.mean(efficiency_scores)
        
        # Weighted combination
        performance_score = (0.4 * avg_utilization + 0.3 * avg_coverage + 0.3 * avg_efficiency) * 100
        
        return min(100.0, max(0.0, performance_score))


if __name__ == "__main__":
    # Example usage of portfolio manager
    portfolio_manager = RIPortfolioManager()
    
    # Sample data
    sample_ris = [
        ReservedInstance(
            ri_id="ri-1234567890abcdef0",
            instance_type="m5.large",
            region="us-east-1",
            availability_zone="us-east-1a",
            term=CommitmentTerm.ONE_YEAR,
            payment_option=PaymentOption.PARTIAL_UPFRONT,
            quantity=2,
            start_date=datetime.now() - timedelta(days=180),
            end_date=datetime.now() + timedelta(days=185),
            hourly_rate=0.069,
            upfront_cost=624.0,
            state="active"
        )
    ]
    
    sample_utilization_reports = [
        UtilizationReport(
            ri_id="ri-1234567890abcdef0",
            reporting_period=(datetime.now() - timedelta(days=30), datetime.now()),
            total_hours_available=1440.0,  # 30 days * 24 hours * 2 instances
            total_hours_used=1008.0,       # 70% utilization
            utilization_percentage=70.0,
            coverage_percentage=85.0,
            unused_hours=432.0,
            cost_savings_realized=150.0,
            cost_savings_potential=200.0,
            efficiency_score=0.75
        )
    ]
    
    sample_savings_analyses = [
        SavingsAnalysis(
            recommendation_id="ri-1234567890abcdef0_1_year",
            total_savings=600.0,
            monthly_savings=50.0,
            percentage_savings=25.0,
            break_even_months=12.5,
            roi_percentage=25.0,
            net_present_value=450.0,
            internal_rate_of_return=22.0,
            payback_period=12.5,
            total_cost_of_ownership=1984.0
        )
    ]
    
    # Generate portfolio overview
    overview = portfolio_manager.generate_portfolio_overview(
        sample_ris, sample_utilization_reports, sample_savings_analyses
    )
    
    print("Portfolio Overview:")
    print(f"  Total RIs: {overview.total_ris}")
    print(f"  Total Investment: ${overview.total_investment:,.2f}")
    print(f"  Annual Savings: ${overview.total_annual_savings:,.2f}")
    print(f"  Portfolio ROI: {overview.portfolio_roi:.1f}%")
    print(f"  Average Utilization: {overview.average_utilization:.1f}%")
    print(f"  Expiring Soon: {overview.expiring_soon}")
    print(f"  Underutilized: {overview.underutilized}")
    print(f"  Risk Score: {overview.risk_score:.2f}")
    
    # Generate modification recommendations
    sample_usage_patterns = [
        UsagePattern(
            resource_id="i-1234567890abcdef0",
            instance_type="m5.large",
            region="us-east-1",
            average_usage=16.8,  # 70% of 24 hours
            peak_usage=24.0,
            minimum_usage=12.0,
            stability_score=0.85,
            predictability_score=0.78,
            seasonal_patterns={"daily": 0.2},
            trend_direction="stable",
            confidence_level=0.82
        )
    ]
    
    modifications = portfolio_manager.generate_modification_recommendations(
        sample_ris, sample_utilization_reports, sample_usage_patterns
    )
    
    print(f"\nModification Recommendations: {len(modifications)}")
    for mod in modifications:
        print(f"  {mod.modification_type.upper()}: {mod.rationale}")
        print(f"    Expected Savings: ${mod.expected_savings:.2f}")
        print(f"    Risk Level: {mod.risk_level}")
    
    # Create capacity plan
    capacity_plan = portfolio_manager.create_capacity_plan(sample_usage_patterns, sample_ris)
    
    print(f"\nCapacity Plan ({capacity_plan.planning_horizon_months} months):")
    print(f"  Projected Growth: {capacity_plan.projected_growth:.1%}")
    print(f"  Budget Requirements: ${capacity_plan.budget_requirements:,.2f}")
    print(f"  Confidence Level: {capacity_plan.confidence_level:.1%}")
    print(f"  Risk Factors: {len(capacity_plan.risk_factors)}")
    for risk in capacity_plan.risk_factors:
        print(f"    - {risk}")
    
    print(f"  Recommended Purchases: {len(capacity_plan.recommended_purchases)}")
    for purchase in capacity_plan.recommended_purchases:
        print(f"    - {purchase['quantity']}x {purchase['instance_type']}: ${purchase['estimated_cost']:,.2f}")