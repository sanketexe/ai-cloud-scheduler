"""
Reinforcement Learning Optimization API Endpoints

This module provides REST API endpoints for the reinforcement learning agent
that continuously improves optimization strategies through trial and feedback.
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
from .reinforcement_learning_agent import (
    ReinforcementLearningAgent, SystemState, OptimizationAction,
    Experience, PolicyUpdateResult, ActionOutcome, RiskLevel
)
from .exceptions import RLAgentError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/rl-optimization", tags=["Reinforcement Learning"])

# Global RL agent instance
_rl_agent: Optional[ReinforcementLearningAgent] = None

def get_rl_agent() -> ReinforcementLearningAgent:
    """Get or create RL agent instance"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = ReinforcementLearningAgent()
    return _rl_agent

# Request/Response Models
class SystemStateRequest(BaseModel):
    """Request model for system state"""
    account_id: str = Field(..., description="Account identifier")
    resource_utilization: Dict[str, float] = Field(..., description="Resource utilization metrics")
    cost_metrics: Dict[str, float] = Field(..., description="Cost metrics")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    external_factors: Dict[str, Any] = Field(default={}, description="External factors")

class ActionSelectionRequest(BaseModel):
    """Request model for action selection"""
    system_state: SystemStateRequest = Field(..., description="Current system state")
    available_actions: List[str] = Field(default=[], description="Available action types")
    constraints: Dict[str, Any] = Field(default={}, description="Action constraints")

class FeedbackRequest(BaseModel):
    """Request model for action feedback"""
    action_id: str = Field(..., description="Action identifier")
    outcome: Dict[str, Any] = Field(..., description="Action outcome")
    reward_components: Dict[str, float] = Field(..., description="Reward components")
    success: bool = Field(..., description="Whether action was successful")
    execution_time: float = Field(..., description="Action execution time in seconds")

class PolicyUpdateRequest(BaseModel):
    """Request model for policy updates"""
    experiences: List[Dict[str, Any]] = Field(..., description="Learning experiences")
    update_type: str = Field(default="incremental", description="Update type")
    learning_rate: Optional[float] = Field(None, description="Learning rate override")

class StrategyComparisonRequest(BaseModel):
    """Request for A/B testing strategy comparison"""
    strategy_a: Dict[str, Any] = Field(..., description="Strategy A configuration")
    strategy_b: Dict[str, Any] = Field(..., description="Strategy B configuration")
    test_duration_hours: int = Field(default=24, description="Test duration in hours")
    traffic_split: float = Field(default=0.5, description="Traffic split for strategy A")

# Response Models
class ActionRecommendationResponse(BaseModel):
    """Response model for action recommendations"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    expected_impact: Dict[str, float]
    risk_level: str
    reasoning: str
    alternatives: List[Dict[str, Any]]

class PolicyPerformanceResponse(BaseModel):
    """Response model for policy performance"""
    policy_version: str
    performance_metrics: Dict[str, float]
    learning_progress: Dict[str, float]
    recent_actions: List[Dict[str, Any]]
    success_rate: float
    average_reward: float

class StrategyComparisonResponse(BaseModel):
    """Response model for strategy comparison"""
    test_id: str
    strategy_a_performance: Dict[str, float]
    strategy_b_performance: Dict[str, float]
    statistical_significance: float
    recommended_strategy: str
    confidence_interval: Dict[str, float]

@router.post("/select-action", response_model=ActionRecommendationResponse)
async def select_optimization_action(
    request: ActionSelectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Select optimal optimization action using RL agent"""
    try:
        logger.info(
            "Selecting optimization action",
            user_id=current_user.id,
            account_id=request.system_state.account_id
        )
        
        agent = get_rl_agent()
        
        # Convert request to system state
        system_state = SystemState(
            timestamp=datetime.now(),
            account_id=request.system_state.account_id,
            resource_utilization=request.system_state.resource_utilization,
            cost_metrics=request.system_state.cost_metrics,
            performance_metrics=request.system_state.performance_metrics,
            external_factors=request.system_state.external_factors
        )
        
        # Select action
        action = await agent.select_action(
            state=system_state,
            available_actions=request.available_actions,
            constraints=request.constraints
        )
        
        # Get action confidence
        confidence = await agent.get_action_confidence(system_state, action)
        
        # Get alternative actions
        alternatives = await agent.get_alternative_actions(
            state=system_state,
            primary_action=action,
            count=3
        )
        
        return ActionRecommendationResponse(
            action_id=action.action_id,
            action_type=action.action_type,
            parameters=action.parameters,
            confidence=confidence,
            expected_impact=action.expected_impact,
            risk_level=action.risk_level.value,
            reasoning=action.reasoning,
            alternatives=[alt.to_dict() for alt in alternatives]
        )
        
    except Exception as e:
        logger.error(f"Action selection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_action_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user)
):
    """Submit feedback for executed action to improve RL policy"""
    try:
        logger.info(
            "Submitting action feedback",
            user_id=current_user.id,
            action_id=request.action_id,
            success=request.success
        )
        
        agent = get_rl_agent()
        
        # Create action outcome
        outcome = ActionOutcome(
            action_id=request.action_id,
            success=request.success,
            outcome_data=request.outcome,
            reward_components=request.reward_components,
            execution_time=request.execution_time,
            timestamp=datetime.now()
        )
        
        # Submit feedback to agent
        reward = await agent.evaluate_action_outcome(outcome)
        
        # Update policy based on feedback
        await agent.update_policy_from_feedback(outcome, reward)
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "action_id": request.action_id,
            "calculated_reward": reward,
            "policy_updated": True
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-policy")
async def update_rl_policy(
    request: PolicyUpdateRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Update RL policy with batch of experiences"""
    try:
        logger.info(
            "Updating RL policy",
            user_id=current_user.id,
            experience_count=len(request.experiences),
            update_type=request.update_type
        )
        
        agent = get_rl_agent()
        
        # Convert experiences
        experiences = []
        for exp_data in request.experiences:
            experience = Experience(
                state=exp_data["state"],
                action=exp_data["action"],
                reward=exp_data["reward"],
                next_state=exp_data["next_state"],
                done=exp_data.get("done", False),
                timestamp=datetime.fromisoformat(exp_data.get("timestamp", datetime.now().isoformat()))
            )
            experiences.append(experience)
        
        # Schedule policy update in background
        background_tasks.add_task(
            _update_policy_background,
            agent,
            experiences,
            request.update_type,
            request.learning_rate
        )
        
        return {
            "success": True,
            "message": "Policy update scheduled",
            "experience_count": len(experiences),
            "update_type": request.update_type
        }
        
    except Exception as e:
        logger.error(f"Policy update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policy-performance", response_model=PolicyPerformanceResponse)
async def get_policy_performance(
    days: int = Query(7, ge=1, le=30, description="Number of days of performance data"),
    current_user: User = Depends(get_current_user)
):
    """Get RL policy performance metrics"""
    try:
        logger.info(
            "Getting policy performance",
            user_id=current_user.id,
            days=days
        )
        
        agent = get_rl_agent()
        
        # Get performance data
        performance = await agent.get_policy_performance(days)
        
        return PolicyPerformanceResponse(
            policy_version=performance["policy_version"],
            performance_metrics=performance["performance_metrics"],
            learning_progress=performance["learning_progress"],
            recent_actions=performance["recent_actions"],
            success_rate=performance["success_rate"],
            average_reward=performance["average_reward"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get policy performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy-comparison", response_model=StrategyComparisonResponse)
async def start_strategy_comparison(
    request: StrategyComparisonRequest,
    current_user: User = Depends(get_current_user)
):
    """Start A/B test comparison between optimization strategies"""
    try:
        logger.info(
            "Starting strategy comparison",
            user_id=current_user.id,
            test_duration_hours=request.test_duration_hours
        )
        
        agent = get_rl_agent()
        
        # Start A/B test
        test_id = await agent.start_strategy_comparison(
            strategy_a=request.strategy_a,
            strategy_b=request.strategy_b,
            test_duration_hours=request.test_duration_hours,
            traffic_split=request.traffic_split,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "test_id": test_id,
            "message": "Strategy comparison started",
            "test_duration_hours": request.test_duration_hours,
            "traffic_split": request.traffic_split
        }
        
    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy-comparison/{test_id}", response_model=StrategyComparisonResponse)
async def get_strategy_comparison_results(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get results of strategy comparison A/B test"""
    try:
        logger.info(
            "Getting strategy comparison results",
            user_id=current_user.id,
            test_id=test_id
        )
        
        agent = get_rl_agent()
        
        # Get comparison results
        results = await agent.get_strategy_comparison_results(test_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Strategy comparison test not found")
        
        return StrategyComparisonResponse(
            test_id=test_id,
            strategy_a_performance=results["strategy_a_performance"],
            strategy_b_performance=results["strategy_b_performance"],
            statistical_significance=results["statistical_significance"],
            recommended_strategy=results["recommended_strategy"],
            confidence_interval=results["confidence_interval"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy comparison results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning-insights")
async def get_learning_insights(
    current_user: User = Depends(get_current_user)
):
    """Get insights about RL agent learning progress and patterns"""
    try:
        logger.info(
            "Getting learning insights",
            user_id=current_user.id
        )
        
        agent = get_rl_agent()
        
        # Get learning insights
        insights = await agent.get_learning_insights()
        
        return {
            "success": True,
            "insights": insights,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get learning insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-policy")
async def reset_rl_policy(
    keep_experiences: bool = Query(True, description="Keep experience replay buffer"),
    current_user: User = Depends(get_current_user)
):
    """Reset RL policy (use with caution)"""
    try:
        logger.info(
            "Resetting RL policy",
            user_id=current_user.id,
            keep_experiences=keep_experiences
        )
        
        agent = get_rl_agent()
        
        # Reset policy
        await agent.reset_policy(keep_experiences=keep_experiences)
        
        return {
            "success": True,
            "message": "RL policy reset successfully",
            "experiences_kept": keep_experiences,
            "reset_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Policy reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def rl_health_check():
    """Health check for RL optimization system"""
    try:
        agent = get_rl_agent()
        health_data = await agent.health_check()
        
        return {
            "status": "healthy",
            "agent_status": health_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"RL health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Helper functions
async def _update_policy_background(
    agent: ReinforcementLearningAgent,
    experiences: List[Experience],
    update_type: str,
    learning_rate: Optional[float]
):
    """Background task for policy updates"""
    try:
        logger.info(f"Starting background policy update with {len(experiences)} experiences")
        
        result = await agent.update_policy(
            experiences=experiences,
            update_type=update_type,
            learning_rate=learning_rate
        )
        
        logger.info(f"Policy update completed: {result}")
        
    except Exception as e:
        logger.error(f"Background policy update failed: {str(e)}")