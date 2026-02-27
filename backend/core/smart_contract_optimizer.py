"""
Smart Contract Optimizer for Reserved Instances

This module provides advanced RI optimization capabilities including:
- SmartContractOptimizer with usage pattern analysis
- ReservedInstanceRecommender with confidence intervals
- CommitmentBalancer for risk vs. discount optimization
- MarketConditionAnalyzer for dynamic reassessment
- Multi-cloud commitment strategy optimization
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import existing components
from .ri_optimization_system import (
    UsagePattern, UsageAnalyzer, RIRecommendation, 
    CommitmentTerm, PaymentOption, CloudProvider
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval for recommendations"""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    methodology: str


@dataclass
class MarketCondition:
    """Market condition data"""
    provider: CloudProvider
    region: str
    instance_type: str
    timestamp: datetime
    pricing_trend: str  # increasing, decreasing, stable
    demand_level: str  # low, medium, high
    availability_score: float
    competitive_pressure: float
    seasonal_factor: float


@dataclass
class RiskProfile:
    """Risk profile for commitment decisions"""
    volatility_score: float
    demand_uncertainty: float
    technology_risk: float
    market_risk: float
    operational_risk: float
    overall_risk_score: float
    risk_category: str  # low, medium, high


@dataclass
class CommitmentStrategy:
    """Multi-cloud commitment strategy"""
    strategy_id: str
    provider_allocation: Dict[CloudProvider, float]
    term_distribution: Dict[CommitmentTerm, float]
    payment_distribution: Dict[PaymentOption, float]
    expected_savings: float
    risk_score: float
    diversification_score: float
    flexibility_score: float


@dataclass
class OptimizationResult:
    """Result of smart contract optimization"""
    recommendations: List[RIRecommendation]
    confidence_intervals: Dict[str, ConfidenceInterval]
    risk_assessment: RiskProfile
    market_analysis: Dict[str, MarketCondition]
    strategy: CommitmentStrategy
    optimization_score: float
    rationale: str


class SmartContractOptimizer:
    """
    Advanced smart contract optimizer that analyzes usage patterns,
    market conditions, and risk factors to provide optimal RI recommendations
    with confidence intervals and multi-cloud strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".SmartContractOptimizer")
        self.usage_analyzer = UsageAnalyzer()
        self.ri_recommender = ReservedInstanceRecommender()
        self.commitment_balancer = CommitmentBalancer()
        self.market_analyzer = MarketConditionAnalyzer()
        
    async def optimize_commitments(self, 
                                 usage_data: List[Any],
                                 budget_constraints: Dict[str, float],
                                 risk_tolerance: str = "medium",
                                 optimization_horizon_months: int = 12) -> OptimizationResult:
        """
        Perform comprehensive commitment optimization across multiple dimensions.
        
        Args:
            usage_data: Historical usage data
            budget_constraints: Budget constraints by provider/region
            risk_tolerance: Risk tolerance level (low, medium, high)
            optimization_horizon_months: Optimization time horizon
            
        Returns:
            Comprehensive optimization result with recommendations and analysis
        """
        self.logger.info("Starting smart contract optimization")
        
        # Analyze usage patterns
        usage_patterns = self.usage_analyzer.analyze_usage_patterns(usage_data)
        
        # Generate recommendations with confidence intervals
        recommendations = await self.ri_recommender.generate_recommendations_with_confidence(
            usage_patterns, risk_tolerance
        )
        
        # Analyze market conditions
        market_conditions = await self.market_analyzer.analyze_current_conditions(
            usage_patterns
        )
        
        # Balance commitments based on risk vs. discount
        balanced_strategy = self.commitment_balancer.optimize_commitment_balance(
            recommendations, market_conditions, risk_tolerance
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            recommendations, usage_patterns, market_conditions
        )
        
        # Assess overall risk
        risk_assessment = self._assess_portfolio_risk(
            recommendations, market_conditions, usage_patterns
        )
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            recommendations, balanced_strategy, risk_assessment
        )
        
        # Generate rationale
        rationale = self._generate_optimization_rationale(
            recommendations, balanced_strategy, market_conditions
        )
        
        result = OptimizationResult(
            recommendations=recommendations,
            confidence_intervals=confidence_intervals,
            risk_assessment=risk_assessment,
            market_analysis={f"{mc.provider.value}_{mc.region}_{mc.instance_type}": mc 
                           for mc in market_conditions},
            strategy=balanced_strategy,
            optimization_score=optimization_score,
            rationale=rationale
        )
        
        self.logger.info(f"Optimization complete with score: {optimization_score:.2f}")
        return result
    
    def _calculate_confidence_intervals(self, 
                                      recommendations: List[RIRecommendation],
                                      usage_patterns: List[UsagePattern],
                                      market_conditions: List[MarketCondition]) -> Dict[str, ConfidenceInterval]:
        """Calculate confidence intervals for recommendations"""
        confidence_intervals = {}
        
        for rec in recommendations:
            # Find corresponding usage pattern
            pattern = next((p for p in usage_patterns if p.resource_id == rec.resource_id), None)
            if not pattern:
                continue
            
            # Calculate confidence based on pattern stability and market conditions
            base_confidence = (pattern.stability_score + pattern.predictability_score) / 2
            
            # Adjust for market volatility
            market_condition = next((mc for mc in market_conditions 
                                   if mc.instance_type == rec.instance_type), None)
            if market_condition:
                market_adjustment = 1.0 - (market_condition.competitive_pressure * 0.2)
                adjusted_confidence = base_confidence * market_adjustment
            else:
                adjusted_confidence = base_confidence
            
            # Calculate bounds for annual savings
            savings_variance = rec.annual_savings * (1.0 - adjusted_confidence) * 0.3
            lower_bound = max(0, rec.annual_savings - savings_variance)
            upper_bound = rec.annual_savings + savings_variance
            
            confidence_intervals[rec.resource_id] = ConfidenceInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=adjusted_confidence,
                methodology="pattern_stability_market_adjusted"
            )
        
        return confidence_intervals
    
    def _assess_portfolio_risk(self, 
                             recommendations: List[RIRecommendation],
                             market_conditions: List[MarketCondition],
                             usage_patterns: List[UsagePattern]) -> RiskProfile:
        """Assess overall portfolio risk"""
        
        # Volatility risk from usage patterns
        stability_scores = [p.stability_score for p in usage_patterns]
        volatility_score = 1.0 - statistics.mean(stability_scores) if stability_scores else 0.5
        
        # Demand uncertainty
        predictability_scores = [p.predictability_score for p in usage_patterns]
        demand_uncertainty = 1.0 - statistics.mean(predictability_scores) if predictability_scores else 0.5
        
        # Technology risk (based on instance types and ages)
        technology_risk = 0.3  # Default moderate risk
        
        # Market risk from market conditions
        if market_conditions:
            competitive_pressures = [mc.competitive_pressure for mc in market_conditions]
            market_risk = statistics.mean(competitive_pressures)
        else:
            market_risk = 0.4  # Default moderate market risk
        
        # Operational risk (based on commitment terms)
        long_term_commitments = len([r for r in recommendations 
                                   if r.recommended_term == CommitmentTerm.THREE_YEAR])
        operational_risk = min(0.8, long_term_commitments / len(recommendations) * 0.6) if recommendations else 0.3
        
        # Calculate overall risk score
        risk_weights = {
            'volatility': 0.25,
            'demand': 0.25,
            'technology': 0.15,
            'market': 0.20,
            'operational': 0.15
        }
        
        overall_risk_score = (
            volatility_score * risk_weights['volatility'] +
            demand_uncertainty * risk_weights['demand'] +
            technology_risk * risk_weights['technology'] +
            market_risk * risk_weights['market'] +
            operational_risk * risk_weights['operational']
        )
        
        # Determine risk category
        if overall_risk_score < 0.3:
            risk_category = "low"
        elif overall_risk_score < 0.6:
            risk_category = "medium"
        else:
            risk_category = "high"
        
        return RiskProfile(
            volatility_score=volatility_score,
            demand_uncertainty=demand_uncertainty,
            technology_risk=technology_risk,
            market_risk=market_risk,
            operational_risk=operational_risk,
            overall_risk_score=overall_risk_score,
            risk_category=risk_category
        )
    
    def _calculate_optimization_score(self, 
                                    recommendations: List[RIRecommendation],
                                    strategy: CommitmentStrategy,
                                    risk_assessment: RiskProfile) -> float:
        """Calculate overall optimization score"""
        if not recommendations:
            return 0.0
        
        # Savings score (normalized)
        total_savings = sum(r.annual_savings for r in recommendations)
        savings_score = min(1.0, total_savings / 10000)  # Normalize to $10k max
        
        # Risk-adjusted score
        risk_adjustment = 1.0 - risk_assessment.overall_risk_score
        
        # Strategy diversification bonus
        diversification_bonus = strategy.diversification_score * 0.1
        
        # Flexibility bonus
        flexibility_bonus = strategy.flexibility_score * 0.1
        
        # Combined score
        optimization_score = (
            savings_score * 0.5 +
            risk_adjustment * 0.3 +
            diversification_bonus +
            flexibility_bonus
        )
        
        return min(1.0, optimization_score) * 100  # Convert to 0-100 scale
    
    def _generate_optimization_rationale(self, 
                                       recommendations: List[RIRecommendation],
                                       strategy: CommitmentStrategy,
                                       market_conditions: List[MarketCondition]) -> str:
        """Generate human-readable optimization rationale"""
        rationale_parts = []
        
        if recommendations:
            total_savings = sum(r.annual_savings for r in recommendations)
            rationale_parts.append(f"Optimized portfolio provides ${total_savings:,.0f} annual savings")
            
            # Term distribution
            one_year_count = len([r for r in recommendations if r.recommended_term == CommitmentTerm.ONE_YEAR])
            three_year_count = len([r for r in recommendations if r.recommended_term == CommitmentTerm.THREE_YEAR])
            
            if one_year_count > three_year_count:
                rationale_parts.append("Favors shorter terms for flexibility")
            else:
                rationale_parts.append("Balances long-term savings with flexibility")
        else:
            rationale_parts.append("No suitable recommendations found for current usage patterns")
        
        # Market conditions
        if market_conditions:
            stable_markets = len([mc for mc in market_conditions if mc.pricing_trend == "stable"])
            if stable_markets > len(market_conditions) * 0.7:
                rationale_parts.append("Stable market conditions support commitment strategy")
            else:
                rationale_parts.append("Market volatility considered in risk assessment")
        
        # Strategy characteristics
        if strategy.diversification_score > 0.7:
            rationale_parts.append("High diversification reduces concentration risk")
        
        # Ensure we always have at least one rationale part
        if not rationale_parts:
            rationale_parts.append("Strategy optimized based on current usage patterns and risk tolerance")
        
        return ". ".join(rationale_parts) + "."


class ReservedInstanceRecommender:
    """
    Advanced RI recommender that provides recommendations with confidence intervals
    and sophisticated risk analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ReservedInstanceRecommender")
        
    async def generate_recommendations_with_confidence(self, 
                                                     usage_patterns: List[UsagePattern],
                                                     risk_tolerance: str = "medium") -> List[RIRecommendation]:
        """
        Generate RI recommendations with confidence intervals and risk assessment.
        
        Args:
            usage_patterns: Analyzed usage patterns
            risk_tolerance: Risk tolerance level
            
        Returns:
            List of RI recommendations with confidence scoring
        """
        self.logger.info(f"Generating recommendations with {risk_tolerance} risk tolerance")
        
        recommendations = []
        
        for pattern in usage_patterns:
            # Generate recommendations for different commitment options
            pattern_recommendations = await self._generate_pattern_recommendations(
                pattern, risk_tolerance
            )
            recommendations.extend(pattern_recommendations)
        
        # Rank by confidence-adjusted savings
        ranked_recommendations = self._rank_by_confidence_adjusted_value(recommendations)
        
        self.logger.info(f"Generated {len(ranked_recommendations)} recommendations")
        return ranked_recommendations
    
    async def _generate_pattern_recommendations(self, 
                                              pattern: UsagePattern,
                                              risk_tolerance: str) -> List[RIRecommendation]:
        """Generate recommendations for a specific usage pattern"""
        recommendations = []
        
        # Determine suitable commitment terms based on risk tolerance
        suitable_terms = self._get_suitable_terms(pattern, risk_tolerance)
        
        for term in suitable_terms:
            for payment in PaymentOption:
                rec = await self._calculate_recommendation_with_confidence(
                    pattern, term, payment, risk_tolerance
                )
                if rec:
                    recommendations.append(rec)
        
        return recommendations
    
    def _get_suitable_terms(self, pattern: UsagePattern, risk_tolerance: str) -> List[CommitmentTerm]:
        """Determine suitable commitment terms based on pattern and risk tolerance"""
        suitable_terms = []
        
        # Always consider 1-year terms
        suitable_terms.append(CommitmentTerm.ONE_YEAR)
        
        # Consider 3-year terms based on stability and risk tolerance
        if pattern.stability_score >= 0.8 and pattern.predictability_score >= 0.7:
            if risk_tolerance in ["medium", "high"]:
                suitable_terms.append(CommitmentTerm.THREE_YEAR)
        elif pattern.stability_score >= 0.9 and risk_tolerance == "high":
            suitable_terms.append(CommitmentTerm.THREE_YEAR)
        
        return suitable_terms
    
    async def _calculate_recommendation_with_confidence(self, 
                                                      pattern: UsagePattern,
                                                      term: CommitmentTerm,
                                                      payment: PaymentOption,
                                                      risk_tolerance: str) -> Optional[RIRecommendation]:
        """Calculate RI recommendation with confidence scoring"""
        try:
            # Base recommendation calculation (simplified)
            recommended_hours = min(pattern.average_usage, pattern.minimum_usage * 1.1)
            quantity = max(1, int(recommended_hours / 24))
            
            # Sample pricing (would use real pricing service)
            pricing_data = self._get_sample_pricing(pattern.instance_type, term, payment)
            if not pricing_data:
                return None
            
            # Calculate costs and savings
            annual_on_demand_cost = pattern.average_usage * 365 * 0.096  # Sample rate
            annual_ri_cost = (pricing_data['upfront'] + pricing_data['hourly'] * 8760) * quantity
            annual_savings = annual_on_demand_cost - annual_ri_cost
            
            if annual_savings <= 0:
                return None
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(pattern, term, payment, risk_tolerance)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(pattern, term, payment)
            
            # ROI and payback calculations
            roi_percentage = (annual_savings / annual_ri_cost) * 100 if annual_ri_cost > 0 else 0
            
            # Calculate payback months based on payment option
            if pricing_data['upfront'] > 0:
                # For upfront payments, calculate time to recover upfront cost
                payback_months = (pricing_data['upfront'] * quantity) / (annual_savings / 12) if annual_savings > 0 else float('inf')
            else:
                # For no upfront payments, payback is immediate (0 months) since there's no upfront investment to recover
                payback_months = 0.0
            
            return RIRecommendation(
                resource_id=pattern.resource_id,
                instance_type=pattern.instance_type,
                region=pattern.region,
                recommended_term=term,
                recommended_payment=payment,
                quantity=quantity,
                annual_savings=annual_savings,
                upfront_cost=pricing_data['upfront'] * quantity,
                monthly_cost=(pricing_data['hourly'] * 8760 * quantity) / 12,
                roi_percentage=roi_percentage,
                payback_months=payback_months,
                risk_score=risk_score,
                confidence_score=confidence_score,
                rationale=self._generate_recommendation_rationale(pattern, term, payment, confidence_score)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating recommendation: {e}")
            return None
    
    def _calculate_confidence_score(self, 
                                  pattern: UsagePattern,
                                  term: CommitmentTerm,
                                  payment: PaymentOption,
                                  risk_tolerance: str) -> float:
        """Calculate confidence score for recommendation"""
        # Base confidence from pattern characteristics
        base_confidence = (pattern.stability_score + pattern.predictability_score) / 2
        
        # Adjust for commitment term
        term_adjustment = 1.0 if term == CommitmentTerm.ONE_YEAR else 0.8
        
        # Adjust for payment option (more upfront = higher risk but potentially higher confidence in commitment)
        payment_adjustments = {
            PaymentOption.NO_UPFRONT: 0.9,
            PaymentOption.PARTIAL_UPFRONT: 1.0,
            PaymentOption.ALL_UPFRONT: 0.95
        }
        payment_adjustment = payment_adjustments.get(payment, 1.0)
        
        # Adjust for risk tolerance
        risk_adjustments = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.1
        }
        risk_adjustment = risk_adjustments.get(risk_tolerance, 1.0)
        
        # Trend adjustment
        trend_adjustment = 1.0
        if pattern.trend_direction == "increasing":
            trend_adjustment = 1.1
        elif pattern.trend_direction == "decreasing":
            trend_adjustment = 0.9
        
        confidence_score = base_confidence * term_adjustment * payment_adjustment * risk_adjustment * trend_adjustment
        
        return min(1.0, max(0.0, confidence_score))
    
    def _calculate_risk_score(self, 
                            pattern: UsagePattern,
                            term: CommitmentTerm,
                            payment: PaymentOption) -> float:
        """Calculate risk score for recommendation"""
        # Pattern-based risk
        pattern_risk = (1.0 - pattern.stability_score) * 0.4 + (1.0 - pattern.predictability_score) * 0.3
        
        # Term risk
        term_risk = 0.2 if term == CommitmentTerm.ONE_YEAR else 0.5
        
        # Payment risk
        payment_risks = {
            PaymentOption.NO_UPFRONT: 0.1,
            PaymentOption.PARTIAL_UPFRONT: 0.3,
            PaymentOption.ALL_UPFRONT: 0.6
        }
        payment_risk = payment_risks.get(payment, 0.3)
        
        # Trend risk
        trend_risk = 0.4 if pattern.trend_direction == "decreasing" else 0.1
        
        total_risk = (pattern_risk + term_risk + payment_risk + trend_risk) / 4
        
        return min(1.0, max(0.0, total_risk))
    
    def _generate_recommendation_rationale(self, 
                                         pattern: UsagePattern,
                                         term: CommitmentTerm,
                                         payment: PaymentOption,
                                         confidence_score: float) -> str:
        """Generate rationale for recommendation"""
        rationale_parts = []
        
        rationale_parts.append(f"Pattern shows {pattern.stability_score:.1%} stability")
        rationale_parts.append(f"and {pattern.predictability_score:.1%} predictability")
        rationale_parts.append(f"with {confidence_score:.1%} confidence in recommendation")
        
        if pattern.trend_direction == "increasing":
            rationale_parts.append("Growing usage supports RI investment")
        elif pattern.trend_direction == "stable":
            rationale_parts.append("Stable usage ideal for commitment")
        
        return ". ".join(rationale_parts) + "."
    
    def _rank_by_confidence_adjusted_value(self, recommendations: List[RIRecommendation]) -> List[RIRecommendation]:
        """Rank recommendations by confidence-adjusted value"""
        scored_recommendations = []
        
        for rec in recommendations:
            # Confidence-adjusted savings
            adjusted_savings = rec.annual_savings * rec.confidence_score
            
            # Risk-adjusted value
            risk_adjusted_value = adjusted_savings * (1.0 - rec.risk_score)
            
            scored_recommendations.append((risk_adjusted_value, rec))
        
        # Sort by adjusted value (descending)
        scored_recommendations.sort(key=lambda x: x[0], reverse=True)
        
        return [rec for score, rec in scored_recommendations]
    
    def _get_sample_pricing(self, instance_type: str, term: CommitmentTerm, payment: PaymentOption) -> Optional[Dict[str, float]]:
        """Get sample pricing data (would use real pricing service)"""
        # Sample pricing data
        pricing_data = {
            "m5.large": {
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
        }
        
        return pricing_data.get(instance_type, {}).get(term, {}).get(payment)


class CommitmentBalancer:
    """
    Balances risk vs. discount optimization for commitment strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".CommitmentBalancer")
        
    def optimize_commitment_balance(self, 
                                  recommendations: List[RIRecommendation],
                                  market_conditions: List[MarketCondition],
                                  risk_tolerance: str) -> CommitmentStrategy:
        """
        Optimize the balance between risk and discount across commitments.
        
        Args:
            recommendations: List of RI recommendations
            market_conditions: Current market conditions
            risk_tolerance: Risk tolerance level
            
        Returns:
            Optimized commitment strategy
        """
        self.logger.info("Optimizing commitment balance")
        
        if not recommendations:
            return self._create_empty_strategy()
        
        # Analyze current recommendation distribution
        provider_distribution = self._analyze_provider_distribution(recommendations)
        term_distribution = self._analyze_term_distribution(recommendations)
        payment_distribution = self._analyze_payment_distribution(recommendations)
        
        # Calculate strategy metrics
        expected_savings = sum(r.annual_savings for r in recommendations)
        base_risk_score = self._calculate_strategy_risk(recommendations, market_conditions)
        diversification_score = self._calculate_diversification_score(recommendations)
        flexibility_score = self._calculate_flexibility_score(recommendations)
        
        # Apply risk tolerance adjustment to final risk score
        risk_tolerance_adjustments = {
            "low": 0.7,    # Cap risk at 70% for low tolerance
            "medium": 0.85, # Cap risk at 85% for medium tolerance  
            "high": 1.0     # No cap for high tolerance
        }
        max_risk_for_tolerance = risk_tolerance_adjustments.get(risk_tolerance, 0.85)
        adjusted_risk_score = min(base_risk_score, max_risk_for_tolerance)
        
        # Optimize based on risk tolerance
        optimized_strategy = self._optimize_for_risk_tolerance(
            recommendations, risk_tolerance, market_conditions
        )
        
        return CommitmentStrategy(
            strategy_id=f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            provider_allocation=provider_distribution,
            term_distribution=term_distribution,
            payment_distribution=payment_distribution,
            expected_savings=expected_savings,
            risk_score=adjusted_risk_score,
            diversification_score=diversification_score,
            flexibility_score=flexibility_score
        )
    
    def _analyze_provider_distribution(self, recommendations: List[RIRecommendation]) -> Dict[CloudProvider, float]:
        """Analyze distribution across cloud providers"""
        # For now, assume all recommendations are AWS (would expand for multi-cloud)
        total_value = sum(r.upfront_cost for r in recommendations)
        
        return {CloudProvider.AWS: 1.0} if total_value > 0 else {}
    
    def _analyze_term_distribution(self, recommendations: List[RIRecommendation]) -> Dict[CommitmentTerm, float]:
        """Analyze distribution across commitment terms"""
        term_counts = defaultdict(int)
        for rec in recommendations:
            term_counts[rec.recommended_term] += 1
        
        total_count = len(recommendations)
        return {term: count / total_count for term, count in term_counts.items()} if total_count > 0 else {}
    
    def _analyze_payment_distribution(self, recommendations: List[RIRecommendation]) -> Dict[PaymentOption, float]:
        """Analyze distribution across payment options"""
        payment_counts = defaultdict(int)
        for rec in recommendations:
            payment_counts[rec.recommended_payment] += 1
        
        total_count = len(recommendations)
        return {payment: count / total_count for payment, count in payment_counts.items()} if total_count > 0 else {}
    
    def _calculate_strategy_risk(self, 
                               recommendations: List[RIRecommendation],
                               market_conditions: List[MarketCondition]) -> float:
        """Calculate overall strategy risk"""
        if not recommendations:
            return 0.0
        
        # Average risk across recommendations
        avg_recommendation_risk = statistics.mean([r.risk_score for r in recommendations])
        
        # Market risk adjustment
        if market_conditions:
            market_volatility = statistics.mean([mc.competitive_pressure for mc in market_conditions])
            market_risk_adjustment = market_volatility * 0.2
        else:
            market_risk_adjustment = 0.0
        
        # Concentration risk (lack of diversification)
        concentration_risk = self._calculate_concentration_risk(recommendations)
        
        # Base total risk
        base_total_risk = avg_recommendation_risk + market_risk_adjustment + concentration_risk
        
        # Cap the risk to ensure it doesn't exceed reasonable bounds
        # This prevents extreme concentration from dominating the risk score
        capped_risk = min(0.9, base_total_risk)
        
        return min(1.0, max(0.0, capped_risk))
    
    def _calculate_diversification_score(self, recommendations: List[RIRecommendation]) -> float:
        """Calculate diversification score"""
        if not recommendations:
            return 0.0
        
        # Instance type diversification
        unique_types = len(set(r.instance_type for r in recommendations))
        type_diversity = min(1.0, unique_types / 5)  # Normalize to max 5 types
        
        # Region diversification
        unique_regions = len(set(r.region for r in recommendations))
        region_diversity = min(1.0, unique_regions / 3)  # Normalize to max 3 regions
        
        # Term diversification
        unique_terms = len(set(r.recommended_term for r in recommendations))
        term_diversity = min(1.0, unique_terms / 2)  # Normalize to max 2 terms
        
        # Combined diversification score
        diversification_score = (type_diversity + region_diversity + term_diversity) / 3
        
        return diversification_score
    
    def _calculate_flexibility_score(self, recommendations: List[RIRecommendation]) -> float:
        """Calculate flexibility score"""
        if not recommendations:
            return 0.0
        
        # Term flexibility (shorter terms = more flexible)
        one_year_ratio = len([r for r in recommendations if r.recommended_term == CommitmentTerm.ONE_YEAR]) / len(recommendations)
        term_flexibility = one_year_ratio
        
        # Payment flexibility (less upfront = more flexible)
        no_upfront_ratio = len([r for r in recommendations if r.recommended_payment == PaymentOption.NO_UPFRONT]) / len(recommendations)
        payment_flexibility = no_upfront_ratio * 0.8 + 0.2  # Base flexibility score
        
        # Combined flexibility score
        flexibility_score = (term_flexibility + payment_flexibility) / 2
        
        return flexibility_score
    
    def _calculate_concentration_risk(self, recommendations: List[RIRecommendation]) -> float:
        """Calculate concentration risk from lack of diversification"""
        if not recommendations:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        # Group by instance type
        type_counts = defaultdict(int)
        for rec in recommendations:
            type_counts[rec.instance_type] += 1
        
        total_count = len(recommendations)
        hhi = sum((count / total_count) ** 2 for count in type_counts.values())
        
        # Convert HHI to risk score (higher HHI = higher concentration = higher risk)
        concentration_risk = (hhi - 0.2) / 0.8 if hhi > 0.2 else 0.0  # Normalize
        
        return min(1.0, max(0.0, concentration_risk))
    
    def _optimize_for_risk_tolerance(self, 
                                   recommendations: List[RIRecommendation],
                                   risk_tolerance: str,
                                   market_conditions: List[MarketCondition]) -> Dict[str, Any]:
        """Optimize strategy based on risk tolerance"""
        optimization_params = {
            "low": {
                "max_three_year_ratio": 0.3,
                "max_all_upfront_ratio": 0.2,
                "min_diversification": 0.7
            },
            "medium": {
                "max_three_year_ratio": 0.6,
                "max_all_upfront_ratio": 0.4,
                "min_diversification": 0.5
            },
            "high": {
                "max_three_year_ratio": 0.8,
                "max_all_upfront_ratio": 0.6,
                "min_diversification": 0.3
            }
        }
        
        params = optimization_params.get(risk_tolerance, optimization_params["medium"])
        
        # Apply optimization constraints
        optimized_recommendations = self._apply_risk_constraints(recommendations, params)
        
        return {
            "optimized_recommendations": optimized_recommendations,
            "constraints_applied": params,
            "risk_tolerance": risk_tolerance
        }
    
    def _apply_risk_constraints(self, 
                              recommendations: List[RIRecommendation],
                              constraints: Dict[str, float]) -> List[RIRecommendation]:
        """Apply risk constraints to recommendations"""
        # For now, return original recommendations
        # In a full implementation, this would filter and adjust recommendations
        # based on the risk constraints
        return recommendations
    
    def _create_empty_strategy(self) -> CommitmentStrategy:
        """Create empty strategy when no recommendations available"""
        return CommitmentStrategy(
            strategy_id="empty_strategy",
            provider_allocation={},
            term_distribution={},
            payment_distribution={},
            expected_savings=0.0,
            risk_score=0.0,
            diversification_score=0.0,
            flexibility_score=0.0
        )


class MarketConditionAnalyzer:
    """
    Analyzes market conditions for dynamic reassessment of RI strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".MarketConditionAnalyzer")
        
    async def analyze_current_conditions(self, usage_patterns: List[UsagePattern]) -> List[MarketCondition]:
        """
        Analyze current market conditions for the given usage patterns.
        
        Args:
            usage_patterns: Usage patterns to analyze market conditions for
            
        Returns:
            List of market conditions for each unique instance type/region combination
        """
        self.logger.info("Analyzing current market conditions")
        
        # Get unique combinations of instance types and regions
        unique_combinations = set((p.instance_type, p.region) for p in usage_patterns)
        
        market_conditions = []
        
        # Analyze conditions for each combination
        for instance_type, region in unique_combinations:
            condition = await self._analyze_instance_market(instance_type, region)
            market_conditions.append(condition)
        
        self.logger.info(f"Analyzed market conditions for {len(market_conditions)} instance/region combinations")
        return market_conditions
    
    async def _analyze_instance_market(self, instance_type: str, region: str) -> MarketCondition:
        """Analyze market conditions for specific instance type and region"""
        
        # Simulate market analysis (in real implementation, would fetch from various sources)
        pricing_trend = self._analyze_pricing_trend(instance_type, region)
        demand_level = self._analyze_demand_level(instance_type, region)
        availability_score = self._calculate_availability_score(instance_type, region)
        competitive_pressure = self._analyze_competitive_pressure(instance_type, region)
        seasonal_factor = self._calculate_seasonal_factor(instance_type, region)
        
        return MarketCondition(
            provider=CloudProvider.AWS,  # Default to AWS for now
            region=region,
            instance_type=instance_type,
            timestamp=datetime.now(),
            pricing_trend=pricing_trend,
            demand_level=demand_level,
            availability_score=availability_score,
            competitive_pressure=competitive_pressure,
            seasonal_factor=seasonal_factor
        )
    
    def _analyze_pricing_trend(self, instance_type: str, region: str) -> str:
        """Analyze pricing trend for instance type in region"""
        # Simulate pricing trend analysis
        # In real implementation, would analyze historical pricing data
        
        trends = ["stable", "increasing", "decreasing"]
        weights = [0.6, 0.25, 0.15]  # Most likely stable
        
        return np.random.choice(trends, p=weights)
    
    def _analyze_demand_level(self, instance_type: str, region: str) -> str:
        """Analyze demand level for instance type in region"""
        # Simulate demand analysis
        # In real implementation, would analyze capacity utilization, spot pricing, etc.
        
        demand_levels = ["low", "medium", "high"]
        weights = [0.2, 0.6, 0.2]  # Most likely medium
        
        return np.random.choice(demand_levels, p=weights)
    
    def _calculate_availability_score(self, instance_type: str, region: str) -> float:
        """Calculate availability score for instance type in region"""
        # Simulate availability scoring
        # In real implementation, would check instance availability, launch success rates, etc.
        
        # Base availability score
        base_score = 0.85
        
        # Adjust based on instance type (newer types might have lower availability)
        if "5" in instance_type:  # Newer generation
            type_adjustment = -0.05
        else:
            type_adjustment = 0.0
        
        # Add some randomness
        random_adjustment = np.random.normal(0, 0.05)
        
        availability_score = base_score + type_adjustment + random_adjustment
        
        return min(1.0, max(0.0, availability_score))
    
    def _analyze_competitive_pressure(self, instance_type: str, region: str) -> float:
        """Analyze competitive pressure in the market"""
        # Simulate competitive pressure analysis
        # In real implementation, would analyze competitor pricing, market share, etc.
        
        # Base competitive pressure
        base_pressure = 0.4
        
        # Adjust based on region (some regions more competitive)
        region_adjustments = {
            "us-east-1": 0.1,  # High competition
            "us-west-2": 0.05,
            "eu-west-1": 0.0,
            "ap-southeast-1": -0.05
        }
        
        region_adjustment = region_adjustments.get(region, 0.0)
        
        # Add randomness
        random_adjustment = np.random.normal(0, 0.1)
        
        competitive_pressure = base_pressure + region_adjustment + random_adjustment
        
        return min(1.0, max(0.0, competitive_pressure))
    
    def _calculate_seasonal_factor(self, instance_type: str, region: str) -> float:
        """Calculate seasonal factor affecting demand/pricing"""
        # Simulate seasonal factor calculation
        # In real implementation, would analyze historical seasonal patterns
        
        current_month = datetime.now().month
        
        # Seasonal patterns (simplified)
        seasonal_multipliers = {
            1: 0.9,   # January - lower demand
            2: 0.9,   # February
            3: 1.0,   # March
            4: 1.1,   # April - spring increase
            5: 1.1,   # May
            6: 1.0,   # June
            7: 0.95,  # July - summer lull
            8: 0.95,  # August
            9: 1.1,   # September - back to school/work
            10: 1.15, # October - peak season
            11: 1.2,  # November - Black Friday, holiday prep
            12: 1.1   # December - holiday season
        }
        
        return seasonal_multipliers.get(current_month, 1.0)


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize the smart contract optimizer
        optimizer = SmartContractOptimizer()
        
        # Sample usage data (would come from real usage analysis)
        sample_usage_patterns = [
            UsagePattern(
                resource_id="i-1234567890abcdef0",
                instance_type="m5.large",
                region="us-east-1",
                average_usage=20.0,
                peak_usage=24.0,
                minimum_usage=16.0,
                stability_score=0.85,
                predictability_score=0.78,
                seasonal_patterns={"daily": 0.2, "weekly": 0.1},
                trend_direction="stable",
                confidence_level=0.82
            ),
            UsagePattern(
                resource_id="i-0987654321fedcba0",
                instance_type="m5.xlarge",
                region="us-west-2",
                average_usage=18.0,
                peak_usage=22.0,
                minimum_usage=14.0,
                stability_score=0.75,
                predictability_score=0.80,
                seasonal_patterns={"daily": 0.15, "weekly": 0.05},
                trend_direction="increasing",
                confidence_level=0.77
            )
        ]
        
        # Run optimization
        result = await optimizer.optimize_commitments(
            usage_data=sample_usage_patterns,  # Would be raw usage data in real implementation
            budget_constraints={"total": 50000.0},
            risk_tolerance="medium",
            optimization_horizon_months=12
        )
        
        # Display results
        print("Smart Contract Optimization Results:")
        print(f"Optimization Score: {result.optimization_score:.2f}")
        print(f"Total Recommendations: {len(result.recommendations)}")
        print(f"Expected Annual Savings: ${sum(r.annual_savings for r in result.recommendations):,.2f}")
        print(f"Overall Risk Score: {result.risk_assessment.overall_risk_score:.2f}")
        print(f"Risk Category: {result.risk_assessment.risk_category}")
        print(f"Strategy Diversification: {result.strategy.diversification_score:.2f}")
        print(f"Strategy Flexibility: {result.strategy.flexibility_score:.2f}")
        print(f"Rationale: {result.rationale}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            confidence_interval = result.confidence_intervals.get(rec.resource_id)
            print(f"{i}. {rec.instance_type} in {rec.region}")
            print(f"   Term: {rec.recommended_term.value}, Payment: {rec.recommended_payment.value}")
            print(f"   Annual Savings: ${rec.annual_savings:,.2f}")
            print(f"   Confidence: {rec.confidence_score:.1%}")
            print(f"   Risk Score: {rec.risk_score:.2f}")
            if confidence_interval:
                print(f"   Savings Range: ${confidence_interval.lower_bound:,.2f} - ${confidence_interval.upper_bound:,.2f}")
            print()
        
        print("Market Conditions:")
        for key, condition in result.market_analysis.items():
            print(f"  {key}: {condition.pricing_trend} pricing, {condition.demand_level} demand")
    
    # Run the example
    asyncio.run(main())