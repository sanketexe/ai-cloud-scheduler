"""
Explainable AI System for Anomaly Detection

Provides human-readable explanations for anomaly detection results,
feature importance analysis, historical context, and actionable recommendations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations"""
    FEATURE_IMPORTANCE = "feature_importance"
    THRESHOLD_VIOLATION = "threshold_violation"
    TREND_ANALYSIS = "trend_analysis"
    SEASONAL_DEVIATION = "seasonal_deviation"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    HISTORICAL_CONTEXT = "historical_context"

class ConfidenceLevel(Enum):
    """Confidence levels for explanations"""
    VERY_HIGH = "very_high"    # 90%+
    HIGH = "high"              # 80-90%
    MEDIUM = "medium"          # 60-80%
    LOW = "low"                # 40-60%
    VERY_LOW = "very_low"      # <40%

class RecommendationPriority(Enum):
    """Priority levels for recommendations"""
    IMMEDIATE = "immediate"     # Act within hours
    HIGH = "high"              # Act within days
    MEDIUM = "medium"          # Act within weeks
    LOW = "low"                # Act when convenient

@dataclass
class FeatureContribution:
    """Individual feature contribution to anomaly"""
    feature_name: str
    importance_score: float
    current_value: float
    baseline_value: float
    deviation_percentage: float
    contribution_explanation: str
    
    def get_impact_description(self) -> str:
        """Get human-readable impact description"""
        if self.deviation_percentage > 50:
            return f"significantly higher than normal ({self.deviation_percentage:.1f}% increase)"
        elif self.deviation_percentage > 20:
            return f"moderately higher than normal ({self.deviation_percentage:.1f}% increase)"
        elif self.deviation_percentage > -20:
            return f"slightly different from normal ({self.deviation_percentage:.1f}% change)"
        elif self.deviation_percentage > -50:
            return f"moderately lower than normal ({abs(self.deviation_percentage):.1f}% decrease)"
        else:
            return f"significantly lower than normal ({abs(self.deviation_percentage):.1f}% decrease)"

@dataclass
class HistoricalContext:
    """Historical context for anomaly"""
    similar_events: List[Dict[str, Any]] = field(default_factory=list)
    seasonal_patterns: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    
    def get_context_summary(self) -> str:
        """Get summary of historical context"""
        context_parts = []
        
        if self.similar_events:
            context_parts.append(f"Found {len(self.similar_events)} similar events in the past")
        
        if self.seasonal_patterns.get('has_seasonality'):
            context_parts.append("Strong seasonal patterns detected")
        
        if self.trend_analysis.get('trend_direction'):
            direction = self.trend_analysis['trend_direction']
            context_parts.append(f"Overall trend is {direction}")
        
        return "; ".join(context_parts) if context_parts else "No significant historical patterns found"

@dataclass
class ActionableRecommendation:
    """Actionable recommendation for addressing anomaly"""
    title: str
    description: str
    priority: RecommendationPriority
    estimated_impact: str
    implementation_effort: str
    timeline: str
    prerequisites: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    
    def get_priority_description(self) -> str:
        """Get human-readable priority description"""
        descriptions = {
            RecommendationPriority.IMMEDIATE: "Requires immediate attention (within hours)",
            RecommendationPriority.HIGH: "High priority (within 1-3 days)",
            RecommendationPriority.MEDIUM: "Medium priority (within 1-2 weeks)",
            RecommendationPriority.LOW: "Low priority (when convenient)"
        }
        return descriptions.get(self.priority, "Unknown priority")

@dataclass
class AnomalyExplanation:
    """Complete explanation for an anomaly"""
    anomaly_id: str
    explanation_id: str
    generated_at: datetime
    
    # Core explanation
    primary_explanation: str
    detailed_explanation: str
    confidence_level: ConfidenceLevel
    explanation_types: List[ExplanationType]
    
    # Feature analysis
    feature_contributions: List[FeatureContribution]
    top_contributing_factors: List[str]
    
    # Context
    historical_context: HistoricalContext
    comparative_analysis: Dict[str, Any]
    
    # Recommendations
    recommendations: List[ActionableRecommendation]
    
    # Metadata
    model_version: str
    explanation_confidence: float
    processing_time_ms: float
    
    def get_summary(self) -> str:
        """Get concise summary of explanation"""
        top_factors = ", ".join(self.top_contributing_factors[:3])
        return f"{self.primary_explanation}. Main factors: {top_factors}."
    
    def get_confidence_description(self) -> str:
        """Get human-readable confidence description"""
        descriptions = {
            ConfidenceLevel.VERY_HIGH: "Very confident in this explanation",
            ConfidenceLevel.HIGH: "Confident in this explanation",
            ConfidenceLevel.MEDIUM: "Moderately confident in this explanation",
            ConfidenceLevel.LOW: "Low confidence in this explanation",
            ConfidenceLevel.VERY_LOW: "Very low confidence - requires manual review"
        }
        return descriptions.get(self.confidence_level, "Unknown confidence")

class ExplanationEngine:
    """
    Explainable AI engine for anomaly detection.
    
    Provides human-readable explanations, feature importance analysis,
    historical context, and actionable recommendations for detected anomalies.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "explanation_data"
        
        # Explanation storage
        self.explanations: Dict[str, AnomalyExplanation] = {}
        self.explanation_templates: Dict[str, str] = {}
        self.recommendation_templates: Dict[str, Dict[str, Any]] = {}
        
        # Feature importance models
        self.feature_importance_weights: Dict[str, float] = {}
        self.baseline_statistics: Dict[str, Dict[str, float]] = {}
        
        # Historical data for context
        self.historical_anomalies: List[Dict[str, Any]] = []
        self.seasonal_patterns: Dict[str, Any] = {}
        
        # Configuration
        self.min_feature_importance = 0.05
        self.max_recommendations = 8
        self.context_window_days = 90
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_explanation_data()
        self._setup_explanation_templates()
        self._setup_recommendation_templates()
    
    async def explain_anomaly(
        self,
        anomaly_id: str,
        anomaly_data: Dict[str, Any],
        model_output: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> AnomalyExplanation:
        """
        Generate comprehensive explanation for an anomaly.
        
        Args:
            anomaly_id: Unique identifier for the anomaly
            anomaly_data: Raw anomaly data and context
            model_output: ML model output including predictions and confidence
            historical_data: Historical data for context analysis
            
        Returns:
            Complete anomaly explanation with recommendations
        """
        start_time = datetime.now()
        explanation_id = f"exp_{anomaly_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Generating explanation {explanation_id} for anomaly {anomaly_id}")
        
        try:
            # Analyze feature contributions
            feature_contributions = await self._analyze_feature_importance(
                anomaly_data, model_output
            )
            
            # Generate primary explanation
            primary_explanation = await self._generate_primary_explanation(
                anomaly_data, feature_contributions
            )
            
            # Generate detailed explanation
            detailed_explanation = await self._generate_detailed_explanation(
                anomaly_data, feature_contributions, model_output
            )
            
            # Determine explanation types
            explanation_types = self._determine_explanation_types(
                anomaly_data, feature_contributions
            )
            
            # Analyze historical context
            historical_context = await self._analyze_historical_context(
                anomaly_data, historical_data
            )
            
            # Generate comparative analysis
            comparative_analysis = await self._generate_comparative_analysis(
                anomaly_data, historical_context
            )
            
            # Generate actionable recommendations
            recommendations = await self._generate_recommendations(
                anomaly_data, feature_contributions, historical_context
            )
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(
                model_output, feature_contributions
            )
            
            # Extract top contributing factors
            top_factors = [
                fc.feature_name for fc in 
                sorted(feature_contributions, key=lambda x: x.importance_score, reverse=True)[:5]
            ]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create explanation
            explanation = AnomalyExplanation(
                anomaly_id=anomaly_id,
                explanation_id=explanation_id,
                generated_at=start_time,
                primary_explanation=primary_explanation,
                detailed_explanation=detailed_explanation,
                confidence_level=confidence_level,
                explanation_types=explanation_types,
                feature_contributions=feature_contributions,
                top_contributing_factors=top_factors,
                historical_context=historical_context,
                comparative_analysis=comparative_analysis,
                recommendations=recommendations,
                model_version=model_output.get('model_version', '1.0'),
                explanation_confidence=model_output.get('confidence', 0.5),
                processing_time_ms=processing_time
            )
            
            # Store explanation
            self.explanations[explanation_id] = explanation
            self._save_explanation(explanation)
            
            # Update historical data
            self._update_historical_data(anomaly_data, explanation)
            
            logger.info(f"Generated explanation {explanation_id} in {processing_time:.1f}ms")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation for anomaly {anomaly_id}: {e}")
            raise
    
    async def _analyze_feature_importance(
        self,
        anomaly_data: Dict[str, Any],
        model_output: Dict[str, Any]
    ) -> List[FeatureContribution]:
        """Analyze feature importance for the anomaly"""
        
        feature_contributions = []
        
        # Get feature importance from model output
        feature_importance = model_output.get('feature_importance', {})
        current_values = anomaly_data.get('features', {})
        baseline_values = anomaly_data.get('baseline_features', {})
        
        # Analyze each feature
        for feature_name, importance in feature_importance.items():
            if importance < self.min_feature_importance:
                continue
            
            current_val = current_values.get(feature_name, 0)
            baseline_val = baseline_values.get(feature_name, current_val)
            
            # Calculate deviation
            if baseline_val != 0:
                deviation_pct = ((current_val - baseline_val) / baseline_val) * 100
            else:
                deviation_pct = 0 if current_val == 0 else 100
            
            # Generate explanation for this feature
            explanation = self._generate_feature_explanation(
                feature_name, current_val, baseline_val, deviation_pct, importance
            )
            
            contribution = FeatureContribution(
                feature_name=feature_name,
                importance_score=importance,
                current_value=current_val,
                baseline_value=baseline_val,
                deviation_percentage=deviation_pct,
                contribution_explanation=explanation
            )
            
            feature_contributions.append(contribution)
        
        # Sort by importance
        feature_contributions.sort(key=lambda x: x.importance_score, reverse=True)
        
        return feature_contributions
    
    def _generate_feature_explanation(
        self,
        feature_name: str,
        current_val: float,
        baseline_val: float,
        deviation_pct: float,
        importance: float
    ) -> str:
        """Generate explanation for individual feature contribution"""
        
        # Map feature names to human-readable descriptions
        feature_descriptions = {
            'cost_amount': 'total cost',
            'instance_count': 'number of instances',
            'cpu_utilization': 'CPU utilization',
            'memory_utilization': 'memory usage',
            'network_traffic': 'network traffic',
            'storage_usage': 'storage consumption',
            'request_count': 'number of requests',
            'execution_time': 'execution duration',
            'data_transfer': 'data transfer volume'
        }
        
        feature_desc = feature_descriptions.get(feature_name, feature_name.replace('_', ' '))
        
        if abs(deviation_pct) < 5:
            return f"The {feature_desc} is within normal range"
        elif deviation_pct > 0:
            return f"The {feature_desc} increased by {deviation_pct:.1f}% from baseline"
        else:
            return f"The {feature_desc} decreased by {abs(deviation_pct):.1f}% from baseline"
    
    async def _generate_primary_explanation(
        self,
        anomaly_data: Dict[str, Any],
        feature_contributions: List[FeatureContribution]
    ) -> str:
        """Generate primary human-readable explanation"""
        
        service = anomaly_data.get('service', 'Unknown service')
        cost_amount = anomaly_data.get('cost_amount', 0)
        baseline_amount = anomaly_data.get('baseline_amount', cost_amount)
        
        if baseline_amount > 0:
            cost_change_pct = ((cost_amount - baseline_amount) / baseline_amount) * 100
        else:
            cost_change_pct = 0
        
        # Determine primary cause
        if feature_contributions:
            primary_factor = feature_contributions[0]
            
            if cost_change_pct > 50:
                return f"{service} costs spiked significantly ({cost_change_pct:.1f}% increase), primarily due to {primary_factor.feature_name.replace('_', ' ')}"
            elif cost_change_pct > 20:
                return f"{service} costs increased moderately ({cost_change_pct:.1f}% increase), mainly caused by {primary_factor.feature_name.replace('_', ' ')}"
            elif cost_change_pct > -20:
                return f"{service} costs show unusual patterns with {primary_factor.feature_name.replace('_', ' ')} as the main factor"
            else:
                return f"{service} costs decreased unexpectedly ({abs(cost_change_pct):.1f}% decrease), primarily due to {primary_factor.feature_name.replace('_', ' ')}"
        else:
            return f"{service} costs show anomalous behavior that requires investigation"
    
    async def _generate_detailed_explanation(
        self,
        anomaly_data: Dict[str, Any],
        feature_contributions: List[FeatureContribution],
        model_output: Dict[str, Any]
    ) -> str:
        """Generate detailed technical explanation"""
        
        explanation_parts = []
        
        # Anomaly overview
        service = anomaly_data.get('service', 'Unknown')
        resource_id = anomaly_data.get('resource_id', 'N/A')
        confidence = model_output.get('confidence', 0)
        
        explanation_parts.append(
            f"Anomaly detected in {service} service (Resource: {resource_id}) "
            f"with {confidence:.1%} confidence."
        )
        
        # Feature analysis
        if feature_contributions:
            explanation_parts.append("Key contributing factors:")
            
            for i, fc in enumerate(feature_contributions[:5], 1):
                explanation_parts.append(
                    f"{i}. {fc.contribution_explanation} "
                    f"(importance: {fc.importance_score:.2f})"
                )
        
        # Model insights
        model_type = model_output.get('model_type', 'ensemble')
        explanation_parts.append(
            f"Analysis performed using {model_type} model with "
            f"{len(feature_contributions)} features analyzed."
        )
        
        # Threshold information
        threshold_info = model_output.get('threshold_info', {})
        if threshold_info:
            explanation_parts.append(
                f"Anomaly score: {threshold_info.get('score', 0):.3f} "
                f"(threshold: {threshold_info.get('threshold', 0):.3f})"
            )
        
        return " ".join(explanation_parts)
    
    def _determine_explanation_types(
        self,
        anomaly_data: Dict[str, Any],
        feature_contributions: List[FeatureContribution]
    ) -> List[ExplanationType]:
        """Determine which types of explanations apply"""
        
        explanation_types = []
        
        # Always include feature importance if we have contributions
        if feature_contributions:
            explanation_types.append(ExplanationType.FEATURE_IMPORTANCE)
        
        # Check for threshold violations
        if anomaly_data.get('threshold_violated'):
            explanation_types.append(ExplanationType.THRESHOLD_VIOLATION)
        
        # Check for trend analysis
        if anomaly_data.get('trend_data'):
            explanation_types.append(ExplanationType.TREND_ANALYSIS)
        
        # Check for seasonal deviations
        if anomaly_data.get('seasonal_data'):
            explanation_types.append(ExplanationType.SEASONAL_DEVIATION)
        
        # Always include comparative analysis
        explanation_types.append(ExplanationType.COMPARATIVE_ANALYSIS)
        
        # Always include historical context
        explanation_types.append(ExplanationType.HISTORICAL_CONTEXT)
        
        return explanation_types
    
    async def _analyze_historical_context(
        self,
        anomaly_data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame]
    ) -> HistoricalContext:
        """Analyze historical context for the anomaly"""
        
        context = HistoricalContext()
        
        # Find similar events
        context.similar_events = self._find_similar_events(anomaly_data)
        
        # Analyze seasonal patterns
        if historical_data is not None and len(historical_data) > 30:
            context.seasonal_patterns = self._analyze_seasonality(historical_data)
        
        # Analyze trends
        if historical_data is not None and len(historical_data) > 7:
            context.trend_analysis = self._analyze_trends(historical_data)
        
        # Baseline comparison
        context.baseline_comparison = self._compare_to_baseline(anomaly_data)
        
        return context
    
    def _find_similar_events(self, anomaly_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar historical events"""
        
        similar_events = []
        current_service = anomaly_data.get('service')
        current_cost = anomaly_data.get('cost_amount', 0)
        
        for historical_anomaly in self.historical_anomalies:
            # Check service match
            if historical_anomaly.get('service') != current_service:
                continue
            
            # Check cost similarity (within 50% range)
            historical_cost = historical_anomaly.get('cost_amount', 0)
            if historical_cost > 0:
                cost_ratio = current_cost / historical_cost
                if 0.5 <= cost_ratio <= 2.0:
                    similar_events.append(historical_anomaly)
        
        # Sort by similarity and return top 5
        return similar_events[:5]
    
    def _analyze_seasonality(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in historical data"""
        
        seasonality = {}
        
        try:
            if 'timestamp' in historical_data.columns and 'cost_amount' in historical_data.columns:
                # Convert timestamp to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(historical_data['timestamp']):
                    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                
                # Extract time components
                historical_data['hour'] = historical_data['timestamp'].dt.hour
                historical_data['day_of_week'] = historical_data['timestamp'].dt.dayofweek
                historical_data['day_of_month'] = historical_data['timestamp'].dt.day
                
                # Analyze hourly patterns
                hourly_avg = historical_data.groupby('hour')['cost_amount'].mean()
                hourly_std = historical_data.groupby('hour')['cost_amount'].std()
                
                seasonality['hourly_patterns'] = {
                    'peak_hour': int(hourly_avg.idxmax()),
                    'low_hour': int(hourly_avg.idxmin()),
                    'variation_coefficient': float(hourly_std.mean() / hourly_avg.mean()) if hourly_avg.mean() > 0 else 0
                }
                
                # Analyze weekly patterns
                weekly_avg = historical_data.groupby('day_of_week')['cost_amount'].mean()
                seasonality['weekly_patterns'] = {
                    'peak_day': int(weekly_avg.idxmax()),
                    'low_day': int(weekly_avg.idxmin()),
                    'weekday_avg': float(weekly_avg[:5].mean()),
                    'weekend_avg': float(weekly_avg[5:].mean())
                }
                
                # Determine if there's strong seasonality
                seasonality['has_seasonality'] = seasonality['hourly_patterns']['variation_coefficient'] > 0.2
                
        except Exception as e:
            logger.error(f"Seasonality analysis failed: {e}")
            seasonality['has_seasonality'] = False
        
        return seasonality
    
    def _analyze_trends(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in historical data"""
        
        trends = {}
        
        try:
            if len(historical_data) >= 7 and 'cost_amount' in historical_data.columns:
                costs = historical_data['cost_amount'].values
                
                # Calculate simple trend
                x = np.arange(len(costs))
                slope = np.polyfit(x, costs, 1)[0]
                
                # Determine trend direction
                if slope > costs.mean() * 0.01:  # 1% of mean per period
                    trend_direction = "increasing"
                elif slope < -costs.mean() * 0.01:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
                
                trends['trend_direction'] = trend_direction
                trends['slope'] = float(slope)
                trends['trend_strength'] = abs(slope) / costs.mean() if costs.mean() > 0 else 0
                
                # Calculate volatility
                trends['volatility'] = float(np.std(costs) / np.mean(costs)) if np.mean(costs) > 0 else 0
                
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            trends['trend_direction'] = "unknown"
        
        return trends
    
    def _compare_to_baseline(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current values to baseline"""
        
        comparison = {}
        
        current_cost = anomaly_data.get('cost_amount', 0)
        baseline_cost = anomaly_data.get('baseline_amount', current_cost)
        
        if baseline_cost > 0:
            comparison['cost_deviation_pct'] = ((current_cost - baseline_cost) / baseline_cost) * 100
            comparison['cost_ratio'] = current_cost / baseline_cost
        else:
            comparison['cost_deviation_pct'] = 0
            comparison['cost_ratio'] = 1
        
        # Categorize deviation
        deviation = abs(comparison['cost_deviation_pct'])
        if deviation > 100:
            comparison['deviation_category'] = "extreme"
        elif deviation > 50:
            comparison['deviation_category'] = "high"
        elif deviation > 20:
            comparison['deviation_category'] = "moderate"
        else:
            comparison['deviation_category'] = "low"
        
        return comparison
    
    async def _generate_comparative_analysis(
        self,
        anomaly_data: Dict[str, Any],
        historical_context: HistoricalContext
    ) -> Dict[str, Any]:
        """Generate comparative analysis"""
        
        analysis = {}
        
        # Compare to similar events
        if historical_context.similar_events:
            similar_costs = [event.get('cost_amount', 0) for event in historical_context.similar_events]
            current_cost = anomaly_data.get('cost_amount', 0)
            
            analysis['similar_events_comparison'] = {
                'count': len(historical_context.similar_events),
                'avg_cost': np.mean(similar_costs),
                'current_vs_similar_avg': (current_cost / np.mean(similar_costs)) if np.mean(similar_costs) > 0 else 1,
                'percentile_rank': self._calculate_percentile_rank(current_cost, similar_costs)
            }
        
        # Compare to seasonal patterns
        if historical_context.seasonal_patterns.get('has_seasonality'):
            analysis['seasonal_comparison'] = {
                'is_peak_time': self._is_peak_time(anomaly_data, historical_context.seasonal_patterns),
                'seasonal_expectation': self._get_seasonal_expectation(anomaly_data, historical_context.seasonal_patterns)
            }
        
        # Compare to trend
        if historical_context.trend_analysis.get('trend_direction'):
            analysis['trend_comparison'] = {
                'aligns_with_trend': self._aligns_with_trend(anomaly_data, historical_context.trend_analysis),
                'trend_deviation': self._calculate_trend_deviation(anomaly_data, historical_context.trend_analysis)
            }
        
        return analysis
    
    def _calculate_percentile_rank(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of value in list"""
        if not values:
            return 50.0
        
        sorted_values = sorted(values)
        rank = sum(1 for v in sorted_values if v <= value)
        return (rank / len(sorted_values)) * 100
    
    def _is_peak_time(self, anomaly_data: Dict[str, Any], seasonal_patterns: Dict[str, Any]) -> bool:
        """Check if anomaly occurred during peak time"""
        # Simplified implementation - would use actual timestamp analysis
        return seasonal_patterns.get('hourly_patterns', {}).get('variation_coefficient', 0) > 0.3
    
    def _get_seasonal_expectation(self, anomaly_data: Dict[str, Any], seasonal_patterns: Dict[str, Any]) -> str:
        """Get seasonal expectation description"""
        if seasonal_patterns.get('has_seasonality'):
            return "Higher costs expected due to seasonal patterns"
        return "No strong seasonal patterns detected"
    
    def _aligns_with_trend(self, anomaly_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> bool:
        """Check if anomaly aligns with historical trend"""
        trend_direction = trend_analysis.get('trend_direction', 'stable')
        cost_change = anomaly_data.get('cost_amount', 0) - anomaly_data.get('baseline_amount', 0)
        
        if trend_direction == 'increasing' and cost_change > 0:
            return True
        elif trend_direction == 'decreasing' and cost_change < 0:
            return True
        elif trend_direction == 'stable':
            return abs(cost_change) < anomaly_data.get('baseline_amount', 0) * 0.1
        
        return False
    
    def _calculate_trend_deviation(self, anomaly_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> float:
        """Calculate deviation from expected trend"""
        # Simplified calculation
        expected_slope = trend_analysis.get('slope', 0)
        actual_change = anomaly_data.get('cost_amount', 0) - anomaly_data.get('baseline_amount', 0)
        
        if expected_slope != 0:
            return abs(actual_change - expected_slope) / abs(expected_slope)
        return 0
    
    async def _generate_recommendations(
        self,
        anomaly_data: Dict[str, Any],
        feature_contributions: List[FeatureContribution],
        historical_context: HistoricalContext
    ) -> List[ActionableRecommendation]:
        """Generate actionable recommendations"""
        
        recommendations = []
        service = anomaly_data.get('service', 'Unknown')
        cost_amount = anomaly_data.get('cost_amount', 0)
        baseline_amount = anomaly_data.get('baseline_amount', cost_amount)
        
        # Cost impact assessment
        cost_impact = cost_amount - baseline_amount
        cost_impact_pct = (cost_impact / baseline_amount * 100) if baseline_amount > 0 else 0
        
        # Generate service-specific recommendations
        service_recommendations = self._get_service_specific_recommendations(
            service, feature_contributions, cost_impact_pct
        )
        recommendations.extend(service_recommendations)
        
        # Generate general recommendations based on cost impact
        if cost_impact_pct > 100:
            recommendations.append(ActionableRecommendation(
                title="Immediate Cost Investigation",
                description="Investigate the root cause of the significant cost increase immediately",
                priority=RecommendationPriority.IMMEDIATE,
                estimated_impact="Prevent further cost escalation",
                implementation_effort="Low - investigation only",
                timeline="Within 2 hours",
                prerequisites=["Access to cost monitoring tools"],
                resources_needed=["DevOps engineer", "Cost monitoring dashboard"],
                success_metrics=["Root cause identified", "Cost trend stabilized"]
            ))
        
        # Generate recommendations based on feature contributions
        for fc in feature_contributions[:3]:  # Top 3 contributors
            feature_recommendations = self._get_feature_specific_recommendations(fc, service)
            recommendations.extend(feature_recommendations)
        
        # Generate recommendations based on historical context
        if historical_context.similar_events:
            recommendations.append(ActionableRecommendation(
                title="Apply Historical Solutions",
                description="Review solutions used for similar past events and apply relevant ones",
                priority=RecommendationPriority.HIGH,
                estimated_impact="Faster resolution based on past experience",
                implementation_effort="Medium - requires analysis of past solutions",
                timeline="Within 24 hours",
                prerequisites=["Access to incident history"],
                resources_needed=["Historical incident data", "Engineering team"],
                success_metrics=["Solution implemented", "Cost normalized"]
            ))
        
        # Limit recommendations and sort by priority
        priority_order = {
            RecommendationPriority.IMMEDIATE: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3
        }
        
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))
        return recommendations[:self.max_recommendations]
    
    def _get_service_specific_recommendations(
        self,
        service: str,
        feature_contributions: List[FeatureContribution],
        cost_impact_pct: float
    ) -> List[ActionableRecommendation]:
        """Get service-specific recommendations"""
        
        recommendations = []
        
        service_templates = {
            'EC2': [
                {
                    'title': 'Review Instance Sizing',
                    'description': 'Analyze if EC2 instances are right-sized for current workload',
                    'priority': RecommendationPriority.HIGH,
                    'timeline': '1-2 days'
                },
                {
                    'title': 'Check Auto Scaling Configuration',
                    'description': 'Verify auto scaling policies are not causing unnecessary scaling',
                    'priority': RecommendationPriority.MEDIUM,
                    'timeline': '2-3 days'
                }
            ],
            'S3': [
                {
                    'title': 'Analyze Storage Classes',
                    'description': 'Review if data is stored in appropriate S3 storage classes',
                    'priority': RecommendationPriority.MEDIUM,
                    'timeline': '3-5 days'
                },
                {
                    'title': 'Check Data Transfer Patterns',
                    'description': 'Investigate unusual data transfer or access patterns',
                    'priority': RecommendationPriority.HIGH,
                    'timeline': '1-2 days'
                }
            ],
            'RDS': [
                {
                    'title': 'Database Performance Analysis',
                    'description': 'Analyze database performance metrics and query patterns',
                    'priority': RecommendationPriority.HIGH,
                    'timeline': '1-2 days'
                },
                {
                    'title': 'Review Backup and Snapshot Costs',
                    'description': 'Check if backup retention policies are causing cost increases',
                    'priority': RecommendationPriority.MEDIUM,
                    'timeline': '2-3 days'
                }
            ],
            'Lambda': [
                {
                    'title': 'Function Execution Analysis',
                    'description': 'Analyze Lambda function execution patterns and duration',
                    'priority': RecommendationPriority.HIGH,
                    'timeline': '1 day'
                },
                {
                    'title': 'Memory Configuration Review',
                    'description': 'Review memory allocation for Lambda functions',
                    'priority': RecommendationPriority.MEDIUM,
                    'timeline': '2-3 days'
                }
            ]
        }
        
        templates = service_templates.get(service, [])
        
        for template in templates:
            # Adjust priority based on cost impact
            priority = template['priority']
            if cost_impact_pct > 50:
                if priority == RecommendationPriority.MEDIUM:
                    priority = RecommendationPriority.HIGH
                elif priority == RecommendationPriority.LOW:
                    priority = RecommendationPriority.MEDIUM
            
            recommendation = ActionableRecommendation(
                title=template['title'],
                description=template['description'],
                priority=priority,
                estimated_impact=f"Potential cost reduction of 10-30%",
                implementation_effort="Medium",
                timeline=template['timeline'],
                prerequisites=[f"Access to {service} console", "Cost monitoring tools"],
                resources_needed=["Cloud engineer", "Monitoring tools"],
                success_metrics=["Cost reduction achieved", "Performance maintained"]
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_feature_specific_recommendations(
        self,
        feature_contribution: FeatureContribution,
        service: str
    ) -> List[ActionableRecommendation]:
        """Get recommendations based on specific feature contributions"""
        
        recommendations = []
        feature_name = feature_contribution.feature_name
        deviation = feature_contribution.deviation_percentage
        
        # Feature-specific recommendations
        if 'cpu' in feature_name.lower() and deviation > 20:
            recommendations.append(ActionableRecommendation(
                title="CPU Utilization Optimization",
                description=f"CPU utilization is {deviation:.1f}% higher than baseline - optimize workload or resize instances",
                priority=RecommendationPriority.HIGH,
                estimated_impact="10-25% cost reduction",
                implementation_effort="Medium",
                timeline="2-3 days",
                prerequisites=["Performance monitoring access"],
                resources_needed=["Performance analysis tools", "Engineering team"],
                success_metrics=["CPU utilization normalized", "Cost per compute hour reduced"]
            ))
        
        elif 'memory' in feature_name.lower() and deviation > 20:
            recommendations.append(ActionableRecommendation(
                title="Memory Usage Optimization",
                description=f"Memory usage is {deviation:.1f}% higher than baseline - investigate memory leaks or resize instances",
                priority=RecommendationPriority.HIGH,
                estimated_impact="15-30% cost reduction",
                implementation_effort="Medium to High",
                timeline="3-5 days",
                prerequisites=["Application monitoring", "Memory profiling tools"],
                resources_needed=["Development team", "Profiling tools"],
                success_metrics=["Memory usage optimized", "Instance right-sizing completed"]
            ))
        
        elif 'storage' in feature_name.lower() and deviation > 30:
            recommendations.append(ActionableRecommendation(
                title="Storage Optimization",
                description=f"Storage usage increased by {deviation:.1f}% - implement data lifecycle policies",
                priority=RecommendationPriority.MEDIUM,
                estimated_impact="20-40% storage cost reduction",
                implementation_effort="Low to Medium",
                timeline="1-2 weeks",
                prerequisites=["Storage access permissions"],
                resources_needed=["Data management tools", "Storage admin"],
                success_metrics=["Storage costs reduced", "Data lifecycle policies implemented"]
            ))
        
        return recommendations
    
    def _determine_confidence_level(
        self,
        model_output: Dict[str, Any],
        feature_contributions: List[FeatureContribution]
    ) -> ConfidenceLevel:
        """Determine confidence level for explanation"""
        
        model_confidence = model_output.get('confidence', 0.5)
        feature_count = len(feature_contributions)
        
        # Calculate overall confidence score
        confidence_score = model_confidence
        
        # Adjust based on feature contributions
        if feature_count >= 3:
            confidence_score += 0.1
        elif feature_count < 2:
            confidence_score -= 0.1
        
        # Adjust based on feature importance distribution
        if feature_contributions:
            top_importance = feature_contributions[0].importance_score
            if top_importance > 0.7:
                confidence_score += 0.1
            elif top_importance < 0.3:
                confidence_score -= 0.1
        
        # Map to confidence levels
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_explanation(self, explanation_id: str) -> Optional[AnomalyExplanation]:
        """Get explanation by ID"""
        return self.explanations.get(explanation_id)
    
    def get_explanations_for_anomaly(self, anomaly_id: str) -> List[AnomalyExplanation]:
        """Get all explanations for a specific anomaly"""
        return [
            exp for exp in self.explanations.values()
            if exp.anomaly_id == anomaly_id
        ]
    
    def get_explanation_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get explanation statistics"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_explanations = [
            exp for exp in self.explanations.values()
            if exp.generated_at > cutoff_date
        ]
        
        if not recent_explanations:
            return {'total_explanations': 0}
        
        # Confidence distribution
        confidence_counts = {}
        for level in ConfidenceLevel:
            confidence_counts[level.value] = len([
                exp for exp in recent_explanations
                if exp.confidence_level == level
            ])
        
        # Average processing time
        avg_processing_time = np.mean([
            exp.processing_time_ms for exp in recent_explanations
        ])
        
        # Most common explanation types
        type_counts = {}
        for exp in recent_explanations:
            for exp_type in exp.explanation_types:
                type_counts[exp_type.value] = type_counts.get(exp_type.value, 0) + 1
        
        # Recommendation statistics
        total_recommendations = sum(len(exp.recommendations) for exp in recent_explanations)
        avg_recommendations = total_recommendations / len(recent_explanations)
        
        return {
            'total_explanations': len(recent_explanations),
            'confidence_distribution': confidence_counts,
            'average_processing_time_ms': avg_processing_time,
            'explanation_type_counts': type_counts,
            'average_recommendations_per_explanation': avg_recommendations,
            'time_period_days': days
        }
    
    def _load_explanation_data(self):
        """Load existing explanation data"""
        
        try:
            # Load explanations
            explanations_file = Path(self.storage_path) / "explanations.json"
            if explanations_file.exists():
                with open(explanations_file, 'r') as f:
                    explanations_data = json.load(f)
                
                for exp_data in explanations_data:
                    explanation = self._deserialize_explanation(exp_data)
                    self.explanations[explanation.explanation_id] = explanation
            
            # Load historical anomalies
            history_file = Path(self.storage_path) / "historical_anomalies.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.historical_anomalies = json.load(f)
            
            logger.info(f"Loaded {len(self.explanations)} explanations and {len(self.historical_anomalies)} historical anomalies")
            
        except Exception as e:
            logger.error(f"Failed to load explanation data: {e}")
    
    def _save_explanation(self, explanation: AnomalyExplanation):
        """Save explanation to storage"""
        
        try:
            # For demo purposes, we'll just log the save operation
            logger.debug(f"Saved explanation {explanation.explanation_id}")
            
        except Exception as e:
            logger.error(f"Failed to save explanation {explanation.explanation_id}: {e}")
    
    def _update_historical_data(self, anomaly_data: Dict[str, Any], explanation: AnomalyExplanation):
        """Update historical data with new anomaly"""
        
        historical_entry = {
            'anomaly_id': explanation.anomaly_id,
            'timestamp': datetime.now().isoformat(),
            'service': anomaly_data.get('service'),
            'cost_amount': anomaly_data.get('cost_amount'),
            'baseline_amount': anomaly_data.get('baseline_amount'),
            'confidence': explanation.explanation_confidence,
            'top_factors': explanation.top_contributing_factors[:3]
        }
        
        self.historical_anomalies.append(historical_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.historical_anomalies) > 1000:
            self.historical_anomalies = self.historical_anomalies[-1000:]
    
    def _setup_explanation_templates(self):
        """Setup explanation templates"""
        
        self.explanation_templates = {
            'cost_spike': "Cost increased significantly due to {primary_factor}",
            'resource_scaling': "Resource scaling caused cost increase in {service}",
            'usage_anomaly': "Unusual usage patterns detected in {service}",
            'configuration_change': "Configuration changes may have caused cost impact"
        }
    
    def _setup_recommendation_templates(self):
        """Setup recommendation templates"""
        
        self.recommendation_templates = {
            'immediate_investigation': {
                'title': 'Immediate Investigation Required',
                'priority': RecommendationPriority.IMMEDIATE,
                'timeline': 'Within 2 hours'
            },
            'resource_optimization': {
                'title': 'Resource Optimization',
                'priority': RecommendationPriority.HIGH,
                'timeline': '1-3 days'
            },
            'cost_monitoring': {
                'title': 'Enhanced Cost Monitoring',
                'priority': RecommendationPriority.MEDIUM,
                'timeline': '1-2 weeks'
            }
        }
    
    def _deserialize_explanation(self, exp_data: Dict[str, Any]) -> AnomalyExplanation:
        """Deserialize explanation from dictionary"""
        
        # This would implement full deserialization
        # For demo purposes, we'll create a simplified version
        return AnomalyExplanation(
            anomaly_id=exp_data['anomaly_id'],
            explanation_id=exp_data['explanation_id'],
            generated_at=datetime.fromisoformat(exp_data['generated_at']),
            primary_explanation=exp_data['primary_explanation'],
            detailed_explanation=exp_data['detailed_explanation'],
            confidence_level=ConfidenceLevel(exp_data['confidence_level']),
            explanation_types=[ExplanationType(t) for t in exp_data['explanation_types']],
            feature_contributions=[],  # Would deserialize feature contributions
            top_contributing_factors=exp_data['top_contributing_factors'],
            historical_context=HistoricalContext(),  # Would deserialize context
            comparative_analysis=exp_data.get('comparative_analysis', {}),
            recommendations=[],  # Would deserialize recommendations
            model_version=exp_data['model_version'],
            explanation_confidence=exp_data['explanation_confidence'],
            processing_time_ms=exp_data['processing_time_ms']
        )

# Global explanation engine instance
explanation_engine = ExplanationEngine()