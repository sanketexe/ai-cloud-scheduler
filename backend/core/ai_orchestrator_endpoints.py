"""
AI Orchestrator API Endpoints

This module provides REST API endpoints for the AI orchestration system,
allowing external systems and UIs to interact with the advanced AI/ML features.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .ai_orchestrator import AIOrchestrator
from .ai_orchestrator_models import OptimizationContext, CoordinatedRecommendation
from .auth import get_current_user

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ai-orchestrator", tags=["AI Orchestrator"])

# Global AI orchestrator instance
ai_orchestrator = None

async def get_ai_orchestrator() -> AIOrchestrator:
    """Get or create AI orchestrator instance"""
    global ai_orchestrator
    if ai_orchestrator is None:
        ai_orchestrator = AIOrchestrator()
        await ai_orchestrator.initialize_systems()
    return ai_orchestrator

# Request/Response Models
class OptimizationRequest(BaseModel):
    """Request model for optimization coordination"""
    account_id: str
    resource_ids: List[str]
    optimization_goals: List[str]
    constraints: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}

class OptimizationResponse(BaseModel):
    """Response model for optimization coordination"""
    coordination_id: str
    primary_recommendation: Dict[str, Any]
    supporting_recommendations: List[Dict[str, Any]]
    coordination_strategy: str
    overall_confidence: float
    combined_impact: Dict[str, float]
    implementation_plan: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]

class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    coordination_id: str
    rating: float  # 0.0 to 1.0
    feedback_text: str = ""

class SystemHealthResponse(BaseModel):
    """Response model for system health"""
    systems: Dict[str, Dict[str, Any]]
    orchestrator: Dict[str, Any]
    timestamp: str

@router.post("/optimize", response_model=OptimizationResponse)
async def coordinate_optimization(
    request: OptimizationRequest,
    current_user: dict = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """
    Coordinate optimization across multiple AI systems.
    
    This endpoint triggers the AI orchestrator to analyze the provided context
    and coordinate recommendations from multiple AI systems.
    """
    try:
        logger.info(f"Optimization request from user {current_user['user_id']} for account {request.account_id}")
        
        # Create optimization context
        context = OptimizationContext(
            user_id=current_user["user_id"],
            account_id=request.account_id,
            resource_ids=request.resource_ids,
            optimization_goals=request.optimization_goals,
            constraints=request.constraints,
            preferences=request.preferences,
            historical_feedback=[]  # Could be populated from database
        )
        
        # Coordinate optimization
        coordinated_recommendation = await orchestrator.coordinate_optimization(context)
        
        # Convert to response format
        response = OptimizationResponse(
            coordination_id=coordinated_recommendation.coordination_id,
            primary_recommendation={
                "system_type": coordinated_recommendation.primary_recommendation.system_type.value,
                "recommendation_id": coordinated_recommendation.primary_recommendation.recommendation_id,
                "resource_id": coordinated_recommendation.primary_recommendation.resource_id,
                "action_type": coordinated_recommendation.primary_recommendation.action_type,
                "confidence": coordinated_recommendation.primary_recommendation.confidence,
                "expected_impact": coordinated_recommendation.primary_recommendation.expected_impact,
                "rationale": coordinated_recommendation.primary_recommendation.rationale,
                "dependencies": coordinated_recommendation.primary_recommendation.dependencies
            },
            supporting_recommendations=[
                {
                    "system_type": rec.system_type.value,
                    "recommendation_id": rec.recommendation_id,
                    "resource_id": rec.resource_id,
                    "action_type": rec.action_type,
                    "confidence": rec.confidence,
                    "expected_impact": rec.expected_impact,
                    "rationale": rec.rationale,
                    "dependencies": rec.dependencies
                }
                for rec in coordinated_recommendation.supporting_recommendations
            ],
            coordination_strategy=coordinated_recommendation.coordination_strategy.value,
            overall_confidence=coordinated_recommendation.overall_confidence,
            combined_impact=coordinated_recommendation.combined_impact,
            implementation_plan=coordinated_recommendation.implementation_plan,
            risk_assessment=coordinated_recommendation.risk_assessment
        )
        
        logger.info(f"Optimization completed with coordination ID {coordinated_recommendation.coordination_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Optimization coordination failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """
    Submit feedback for a coordinated recommendation.
    
    This endpoint allows users to provide feedback on the effectiveness
    of AI orchestrator recommendations.
    """
    try:
        logger.info(f"Feedback submission from user {current_user['user_id']} for coordination {request.coordination_id}")
        
        # Validate rating
        if not 0.0 <= request.rating <= 1.0:
            raise HTTPException(status_code=400, detail="Rating must be between 0.0 and 1.0")
        
        # Record feedback in the recommendation engine
        await orchestrator.recommendation_engine.record_feedback(
            user_id=current_user["user_id"],
            coordination_id=request.coordination_id,
            rating=request.rating,
            feedback_text=request.feedback_text
        )
        
        logger.info(f"Feedback recorded successfully for coordination {request.coordination_id}")
        
        return {"message": "Feedback recorded successfully", "coordination_id": request.coordination_id}
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: dict = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """
    Get health status of all AI systems.
    
    This endpoint provides comprehensive health information about
    all AI systems managed by the orchestrator.
    """
    try:
        logger.info(f"Health check request from user {current_user['user_id']}")
        
        # Get system health
        health_data = await orchestrator.get_system_health()
        
        response = SystemHealthResponse(
            systems={k: v for k, v in health_data.items() if k != "orchestrator"},
            orchestrator=health_data.get("orchestrator", {}),
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/user-insights")
async def get_user_insights(
    current_user: dict = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """
    Get personalized insights for the current user.
    
    This endpoint provides insights about user preferences,
    interaction patterns, and recommendation effectiveness.
    """
    try:
        logger.info(f"User insights request from user {current_user['user_id']}")
        
        # Get user insights from recommendation engine
        insights = await orchestrator.recommendation_engine.get_user_insights(
            current_user["user_id"]
        )
        
        return {
            "user_insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"User insights request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User insights failed: {str(e)}")

@router.get("/coordination-history")
async def get_coordination_history(
    account_id: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """
    Get coordination history for the user.
    
    This endpoint provides historical information about
    AI orchestrator coordinations and their outcomes.
    """
    try:
        logger.info(f"Coordination history request from user {current_user['user_id']}")
        
        # Get user profile to access interaction history
        user_profile = await orchestrator.recommendation_engine._get_user_profile(
            current_user["user_id"]
        )
        
        interaction_history = user_profile.get("interaction_history", [])
        
        # Filter by account if specified
        if account_id:
            interaction_history = [
                interaction for interaction in interaction_history
                if interaction.get("account_id") == account_id
            ]
        
        # Apply limit
        interaction_history = interaction_history[-limit:]
        
        return {
            "coordination_history": interaction_history,
            "total_count": len(interaction_history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Coordination history request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Coordination history failed: {str(e)}")

@router.post("/systems/{system_type}/status")
async def update_system_status(
    system_type: str,
    status_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """
    Update status for a specific AI system.
    
    This endpoint allows individual AI systems to report
    their status and performance metrics.
    """
    try:
        logger.info(f"System status update for {system_type} from user {current_user['user_id']}")
        
        # Validate system type
        valid_systems = [system.value for system in orchestrator.ai_systems.keys()]
        if system_type not in valid_systems:
            raise HTTPException(status_code=400, detail=f"Invalid system type: {system_type}")
        
        # Update system status (simplified - in production this would be more sophisticated)
        logger.info(f"Status update received for {system_type}: {status_data}")
        
        return {
            "message": f"Status updated for {system_type}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")

@router.get("/recommendations/{coordination_id}")
async def get_recommendation_details(
    coordination_id: str,
    current_user: dict = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """
    Get detailed information about a specific coordination.
    
    This endpoint provides comprehensive details about
    a specific AI orchestrator coordination.
    """
    try:
        logger.info(f"Recommendation details request for {coordination_id} from user {current_user['user_id']}")
        
        # Get user profile to find the coordination
        user_profile = await orchestrator.recommendation_engine._get_user_profile(
            current_user["user_id"]
        )
        
        # Find the specific coordination
        coordination = None
        for interaction in user_profile.get("interaction_history", []):
            if interaction.get("coordination_id") == coordination_id:
                coordination = interaction
                break
        
        if not coordination:
            raise HTTPException(status_code=404, detail="Coordination not found")
        
        return {
            "coordination": coordination,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation details request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")