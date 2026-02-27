"""
Smart Contract Optimizer API Endpoints

This module provides REST API endpoints for the smart contract optimizer
that optimizes reserved instance and savings plan purchases using AI.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import uuid

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from .auth import get_current_user
from .models import User
from .smart_contract_optimizer import (
    SmartContractOptimizer, UsagePattern, ReservedInstanceRecommendation,
    CommitmentStrategy, MarketCondition, CommitmentType, RiskTolerance
)
from .exceptions import OptimizerError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/smart-contracts", tags=["Smart Contract Optimizer"])

# Global optimizer instance
_optimizer: Optional[SmartContractOptimizer] = None

def get_optimizer() -> SmartContractOptimizer:
    """Get or create smart contract optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = SmartContractOptimizer()
    return _optimizer

# Request/Response Models
class UsageAnalysisRequest(BaseModel):
    """Request model for usage pattern analysis"""
    account_id: str = Field(..., description="Account identifier")
    resource_types: List[str] = Field(..., description="Resource types to analyze")
    analysis_period_days: int = Field(default=90, ge=30, le=365, description="Analysis period in days")
    include_seasonal_patterns: bool = Field(default=True, description="Include seasonal analysis")

class RIRecommendationRequest(BaseModel):
    """Request model for RI recommendations"""
    usage_patterns: Dict[str, Any] = Field(..., description="Usage patterns data")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance level")
    commitment_preferences: Dict[str, Any] = Field(default={}, description="Commitment preferences")
    budget_constraints: Optional[Dict[str, float]] = Field(None, description="Budget constraints")

class CommitmentOptimizationRequest(BaseModel):
    """Request model for commitment optimization"""
    current_commitments: List[Dict[str, Any]] = Field(..., description="Current commitments")
    usage_forecasts: Dict[str, Any] = Field(..., description="Usage forecasts")
    optimization_goals: List[str] = Field(default=["cost", "flexibility"], description="Optimization goals")
    time_horizon_months: int = Field(default=12, ge=6, le=36, description="Optimization time horizon")

class MarketAnalysisRequest(BaseModel):
    """Request model for market condition analysis"""
    providers: List[str] = Field(..., description="Cloud providers to analyze")
    services: List[str] = Field(..., description="Services to analyze")
    regions: List[str] = Field(..., description="Regions to analyze")

class CommitmentRebalanceRequest(BaseModel):
    """Request model for commitment rebalancing"""
    portfolio_id: str = Field(..., description="Commitment portfolio identifier")
    rebalance_triggers: List[str] = Field(..., description="Rebalancing triggers")
    constraints: Dict[str, Any] = Field(default={}, description="Rebalancing constraints")

# Response Models
class UsagePatternResponse(BaseModel):
    """Response model for usage patterns"""
    account_id: str
    resource_type: str
    usage_statistics: Dict[str, float]
    seasonal_patterns: Dict[str, Any]
    growth_trends: Dict[str, float]
    variability_metrics: Dict[str, float]
    recommendations: List[str]

class RIRecommendationResponse(BaseModel):
    """Response model for RI recommendations"""
    recommendation_id: str
    resource_type: str
    instance_family: str
    commitment_type: str
    commitment_term: int
    upfront_payment: float
    monthly_payment: float
    estimated_savings: float
    confidence_score: float
    risk_assessment: Dict[str, Any]
    utilization_forecast: Dict[str, float]

class CommitmentPortfolioResponse(BaseModel):
    """Response model for commitment portfolio"""
    portfolio_id: str
    total_commitments: float
    total_savings: float
    utilization_rate: float
    risk_score: float
    recommendations: List[Dict[str, Any]]
    rebalancing_opportunities: List[Dict[str, Any]]

class MarketInsightsResponse(BaseModel):
    """Response model for market insights"""
    market_conditions: Dict[str, Any]
    pricing_trends: Dict[str, Any]
    discount_opportunities: List[Dict[str, Any]]
    timing_recommendations: Dict[str, str]
    risk_factors: List[str]

@router.post("/analyze-usage", response_model=List[UsagePatternResponse])
async def analyze_usage_patterns(
    request: UsageAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze usage patterns for commitment optimization"""
    try:
        logger.info(
            "Analyzing usage patterns",
            user_id=current_user.id,
            account_id=request.account_id,
            resource_types=request.resource_types
        )
        
        optimizer = get_optimizer()
        
        # Analyze usage patterns for each resource type
        usage_patterns = []
        for resource_type in request.resource_types:
            pattern = await optimizer.analyze_usage_pattern(
                account_id=request.account_id,
                resource_type=resource_type,
                analysis_period_days=request.analysis_period_days,
                include_seasonal_patterns=request.include_seasonal_patterns
            )
            
            usage_patterns.append(UsagePatternResponse(
                account_id=request.account_id,
                resource_type=resource_type,
                usage_statistics=pattern.usage_statistics,
                seasonal_patterns=pattern.seasonal_patterns,
                growth_trends=pattern.growth_trends,
                variability_metrics=pattern.variability_metrics,
                recommendations=pattern.recommendations
            ))
        
        return usage_patterns
        
    except Exception as e:
        logger.error(f"Usage pattern analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ri-recommendations", response_model=List[RIRecommendationResponse])
async def get_ri_recommendations(
    request: RIRecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """Get reserved instance recommendations"""
    try:
        logger.info(
            "Getting RI recommendations",
            user_id=current_user.id,
            risk_tolerance=request.risk_tolerance
        )
        
        optimizer = get_optimizer()
        
        # Convert risk tolerance
        risk_tolerance = RiskTolerance(request.risk_tolerance)
        
        # Get recommendations
        recommendations = await optimizer.get_ri_recommendations(
            usage_patterns=request.usage_patterns,
            risk_tolerance=risk_tolerance,
            commitment_preferences=request.commitment_preferences,
            budget_constraints=request.budget_constraints
        )
        
        # Convert to response format
        response_recommendations = []
        for rec in recommendations:
            response_recommendations.append(RIRecommendationResponse(
                recommendation_id=rec.recommendation_id,
                resource_type=rec.resource_type,
                instance_family=rec.instance_family,
                commitment_type=rec.commitment_type.value,
                commitment_term=rec.commitment_term,
                upfront_payment=float(rec.upfront_payment),
                monthly_payment=float(rec.monthly_payment),
                estimated_savings=float(rec.estimated_savings),
                confidence_score=rec.confidence_score,
                risk_assessment=rec.risk_assessment,
                utilization_forecast=rec.utilization_forecast
            ))
        
        return response_recommendations
        
    except Exception as e:
        logger.error(f"RI recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-commitments", response_model=CommitmentPortfolioResponse)
async def optimize_commitment_portfolio(
    request: CommitmentOptimizationRequest,
    current_user: User = Depends(get_current_user)
):
    """Optimize commitment portfolio across multiple providers"""
    try:
        logger.info(
            "Optimizing commitment portfolio",
            user_id=current_user.id,
            commitment_count=len(request.current_commitments),
            time_horizon=request.time_horizon_months
        )
        
        optimizer = get_optimizer()
        
        # Optimize portfolio
        portfolio = await optimizer.optimize_commitment_portfolio(
            current_commitments=request.current_commitments,
            usage_forecasts=request.usage_forecasts,
            optimization_goals=request.optimization_goals,
            time_horizon_months=request.time_horizon_months
        )
        
        return CommitmentPortfolioResponse(
            portfolio_id=portfolio.portfolio_id,
            total_commitments=float(portfolio.total_commitments),
            total_savings=float(portfolio.total_savings),
            utilization_rate=portfolio.utilization_rate,
            risk_score=portfolio.risk_score,
            recommendations=[rec.to_dict() for rec in portfolio.recommendations],
            rebalancing_opportunities=[opp.to_dict() for opp in portfolio.rebalancing_opportunities]
        )
        
    except Exception as e:
        logger.error(f"Commitment optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-analysis", response_model=MarketInsightsResponse)
async def get_market_analysis(
    providers: List[str] = Query(..., description="Cloud providers"),
    services: List[str] = Query(..., description="Services to analyze"),
    regions: List[str] = Query(..., description="Regions to analyze"),
    current_user: User = Depends(get_current_user)
):
    """Get market analysis and pricing insights"""
    try:
        logger.info(
            "Getting market analysis",
            user_id=current_user.id,
            providers=providers,
            services=services,
            regions=regions
        )
        
        optimizer = get_optimizer()
        
        # Get market analysis
        market_insights = await optimizer.analyze_market_conditions(
            providers=providers,
            services=services,
            regions=regions
        )
        
        return MarketInsightsResponse(
            market_conditions=market_insights.market_conditions,
            pricing_trends=market_insights.pricing_trends,
            discount_opportunities=[opp.to_dict() for opp in market_insights.discount_opportunities],
            timing_recommendations=market_insights.timing_recommendations,
            risk_factors=market_insights.risk_factors
        )
        
    except Exception as e:
        logger.error(f"Market analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebalance-portfolio")
async def rebalance_commitment_portfolio(
    request: CommitmentRebalanceRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Rebalance commitment portfolio based on triggers"""
    try:
        logger.info(
            "Rebalancing commitment portfolio",
            user_id=current_user.id,
            portfolio_id=request.portfolio_id,
            triggers=request.rebalance_triggers
        )
        
        optimizer = get_optimizer()
        
        # Schedule rebalancing in background
        background_tasks.add_task(
            _rebalance_portfolio_background,
            optimizer,
            request.portfolio_id,
            request.rebalance_triggers,
            request.constraints,
            str(current_user.id)
        )
        
        return {
            "success": True,
            "message": "Portfolio rebalancing initiated",
            "portfolio_id": request.portfolio_id,
            "triggers": request.rebalance_triggers,
            "initiated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio rebalancing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/{portfolio_id}/status")
async def get_portfolio_status(
    portfolio_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get current status of commitment portfolio"""
    try:
        logger.info(
            "Getting portfolio status",
            user_id=current_user.id,
            portfolio_id=portfolio_id
        )
        
        optimizer = get_optimizer()
        
        # Get portfolio status
        status = await optimizer.get_portfolio_status(portfolio_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return {
            "success": True,
            "portfolio_status": status,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/utilization-tracking/{commitment_id}")
async def track_commitment_utilization(
    commitment_id: str,
    days: int = Query(30, ge=7, le=90, description="Number of days to track"),
    current_user: User = Depends(get_current_user)
):
    """Track utilization of specific commitment"""
    try:
        logger.info(
            "Tracking commitment utilization",
            user_id=current_user.id,
            commitment_id=commitment_id,
            days=days
        )
        
        optimizer = get_optimizer()
        
        # Get utilization data
        utilization = await optimizer.track_commitment_utilization(
            commitment_id=commitment_id,
            days=days
        )
        
        return {
            "success": True,
            "commitment_id": commitment_id,
            "utilization_data": utilization,
            "tracking_period_days": days,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Utilization tracking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/savings-forecast")
async def forecast_commitment_savings(
    commitment_scenarios: List[Dict[str, Any]],
    forecast_months: int = Query(12, ge=6, le=36, description="Forecast period in months"),
    current_user: User = Depends(get_current_user)
):
    """Forecast savings for different commitment scenarios"""
    try:
        logger.info(
            "Forecasting commitment savings",
            user_id=current_user.id,
            scenario_count=len(commitment_scenarios),
            forecast_months=forecast_months
        )
        
        optimizer = get_optimizer()
        
        # Forecast savings for each scenario
        forecasts = []
        for scenario in commitment_scenarios:
            forecast = await optimizer.forecast_commitment_savings(
                scenario=scenario,
                forecast_months=forecast_months
            )
            forecasts.append(forecast)
        
        return {
            "success": True,
            "savings_forecasts": forecasts,
            "scenario_count": len(commitment_scenarios),
            "forecast_months": forecast_months,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Savings forecast failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization-opportunities")
async def get_optimization_opportunities(
    account_id: str = Query(..., description="Account identifier"),
    min_savings_threshold: float = Query(100.0, description="Minimum savings threshold"),
    current_user: User = Depends(get_current_user)
):
    """Get current optimization opportunities"""
    try:
        logger.info(
            "Getting optimization opportunities",
            user_id=current_user.id,
            account_id=account_id,
            min_savings_threshold=min_savings_threshold
        )
        
        optimizer = get_optimizer()
        
        # Get opportunities
        opportunities = await optimizer.get_optimization_opportunities(
            account_id=account_id,
            min_savings_threshold=min_savings_threshold
        )
        
        return {
            "success": True,
            "opportunities": [opp.to_dict() for opp in opportunities],
            "opportunity_count": len(opportunities),
            "total_potential_savings": sum(opp.estimated_savings for opp in opportunities),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def smart_contract_health_check():
    """Health check for smart contract optimizer"""
    try:
        optimizer = get_optimizer()
        health_data = await optimizer.health_check()
        
        return {
            "status": "healthy",
            "optimizer_status": health_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Smart contract health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Helper functions
async def _rebalance_portfolio_background(
    optimizer: SmartContractOptimizer,
    portfolio_id: str,
    triggers: List[str],
    constraints: Dict[str, Any],
    user_id: str
):
    """Background task for portfolio rebalancing"""
    try:
        logger.info(f"Starting background portfolio rebalancing for {portfolio_id}")
        
        result = await optimizer.rebalance_portfolio(
            portfolio_id=portfolio_id,
            triggers=triggers,
            constraints=constraints,
            user_id=user_id
        )
        
        logger.info(f"Portfolio rebalancing completed: {result}")
        
    except Exception as e:
        logger.error(f"Background portfolio rebalancing failed: {str(e)}")