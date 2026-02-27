"""
API Endpoints for Collaborative Budget Planner

This module provides REST API endpoints for the collaborative budget planning
functionality including multi-user editing, forecasting, and scenario management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from .database import get_db_session
from .auth import get_current_user
from .collaborative_budget_planner import (
    collaborative_budget_planner, BudgetEditType, ForecastPeriod, ScenarioType
)
from .collaborative_forecasting_engine import (
    collaborative_forecasting_engine, ForecastInput, ForecastMethod
)
from .collaborative_session_manager import session_manager
from .models import User

logger = logging.getLogger(__name__)

# Pydantic models for request/response

class BudgetConfigRequest(BaseModel):
    """Request model for creating a collaborative budget"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    total_amount: float = Field(..., gt=0)
    currency: str = Field("USD", max_length=3)
    period: str = Field("monthly")
    categories: List[Dict[str, Any]] = Field(default_factory=list)
    created_by: str

class BudgetEditRequest(BaseModel):
    """Request model for editing budget items"""
    item_path: str = Field(..., min_length=1)
    changes: Dict[str, Any]
    edit_type: str = Field("update")

class CategoryAddRequest(BaseModel):
    """Request model for adding budget categories"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    amount: float = Field(..., ge=0)

class LineItemAddRequest(BaseModel):
    """Request model for adding line items"""
    category_id: str
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    amount: float = Field(..., ge=0)
    confidence_level: float = Field(1.0, ge=0.0, le=1.0)

class ForecastInputRequest(BaseModel):
    """Request model for forecast input"""
    forecast_target: str
    target_path: str
    period: str = Field("monthly")
    period_start: datetime
    period_end: datetime
    estimated_value: float = Field(..., ge=0)
    confidence_level: float = Field(1.0, ge=0.0, le=1.0)
    reasoning: str = Field("", max_length=2000)
    assumptions: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    methodology: str = Field("", max_length=500)
    supporting_data: Dict[str, Any] = Field(default_factory=dict)

class ScenarioCreateRequest(BaseModel):
    """Request model for creating scenarios"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    scenario_type: str = Field("realistic")
    assumptions: List[str] = Field(default_factory=list)

class ScenarioCompareRequest(BaseModel):
    """Request model for comparing scenarios"""
    scenario_ids: List[str] = Field(..., min_items=2)
    comparison_name: str = Field(..., min_length=1, max_length=200)

# API Router
router = APIRouter(prefix="/api/collaborative-budget", tags=["Collaborative Budget"])

@router.post("/sessions/{session_id}/budget")
async def create_collaborative_budget(
    session_id: str,
    budget_config: BudgetConfigRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new collaborative budget for a session"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Create budget
        budget = await collaborative_budget_planner.create_budget_session(
            session_id=session_id,
            budget_config=budget_config.dict(),
            participants=participant_ids
        )
        
        return {
            "success": True,
            "budget": {
                "budget_id": budget.budget_id,
                "name": budget.name,
                "description": budget.description,
                "total_amount": str(budget.total_amount),
                "currency": budget.currency,
                "period": budget.period.value,
                "categories": len(budget.categories),
                "version": budget.version
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating collaborative budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/budget")
async def get_budget_state(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get current collaborative budget state"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Get budget state
        budget_state = await collaborative_budget_planner.get_budget_state(session_id)
        
        if not budget_state:
            raise HTTPException(status_code=404, detail="No active budget found for session")
        
        return {
            "success": True,
            **budget_state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting budget state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/sessions/{session_id}/budget/edit")
async def edit_budget_item(
    session_id: str,
    edit_request: BudgetEditRequest,
    current_user: User = Depends(get_current_user)
):
    """Edit a budget item with conflict detection"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Edit budget item
        result = await collaborative_budget_planner.edit_budget_item(
            session_id=session_id,
            item_path=edit_request.item_path,
            changes=edit_request.changes,
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error editing budget item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/budget/categories")
async def add_budget_category(
    session_id: str,
    category_request: CategoryAddRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a new budget category"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Add category
        result = await collaborative_budget_planner.add_budget_category(
            session_id=session_id,
            category_config=category_request.dict(),
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding budget category: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/budget/line-items")
async def add_line_item(
    session_id: str,
    item_request: LineItemAddRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a new line item to a budget category"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Add line item
        result = await collaborative_budget_planner.add_line_item(
            session_id=session_id,
            category_id=item_request.category_id,
            item_config=item_request.dict(exclude={"category_id"}),
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding line item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecasting endpoints

@router.post("/sessions/{session_id}/forecasts/input")
async def submit_forecast_input(
    session_id: str,
    forecast_request: ForecastInputRequest,
    current_user: User = Depends(get_current_user)
):
    """Submit forecast input for collaborative forecasting"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Create forecast input
        forecast_input = ForecastInput(
            input_id=f"forecast_{session_id}_{current_user.id}_{int(datetime.utcnow().timestamp())}",
            session_id=session_id,
            user_id=str(current_user.id),
            forecast_target=forecast_request.forecast_target,
            target_path=forecast_request.target_path,
            period=ForecastPeriod(forecast_request.period),
            period_start=forecast_request.period_start,
            period_end=forecast_request.period_end,
            estimated_value=Decimal(str(forecast_request.estimated_value)),
            confidence_level=forecast_request.confidence_level,
            reasoning=forecast_request.reasoning,
            assumptions=forecast_request.assumptions,
            risk_factors=forecast_request.risk_factors,
            methodology=forecast_request.methodology,
            supporting_data=forecast_request.supporting_data
        )
        
        # Submit forecast input
        result = await collaborative_forecasting_engine.collect_forecast_input(
            session_id=session_id,
            forecast_input=forecast_input
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error submitting forecast input: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/forecasts")
async def get_forecast_state(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get current forecasting state for a session"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Get forecast state
        forecast_state = await collaborative_forecasting_engine.get_forecast_state(session_id)
        
        return {
            "success": True,
            **forecast_state
        }
        
    except Exception as e:
        logger.error(f"Error getting forecast state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scenario management endpoints

@router.post("/sessions/{session_id}/scenarios")
async def create_scenario(
    session_id: str,
    scenario_request: ScenarioCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new budget scenario"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Create scenario
        scenario_config = scenario_request.dict()
        scenario_config["created_by"] = str(current_user.id)
        
        result = await collaborative_forecasting_engine.create_scenario(
            session_id=session_id,
            scenario_config=scenario_config
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/scenarios/compare")
async def compare_scenarios(
    session_id: str,
    compare_request: ScenarioCompareRequest,
    current_user: User = Depends(get_current_user)
):
    """Compare multiple budget scenarios"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Compare scenarios
        result = await collaborative_forecasting_engine.compare_scenarios(
            session_id=session_id,
            scenario_ids=compare_request.scenario_ids,
            comparison_name=compare_request.comparison_name,
            created_by=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error comparing scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/scenarios")
async def get_scenarios(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all scenarios for a session"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Get forecast state (includes scenarios)
        forecast_state = await collaborative_forecasting_engine.get_forecast_state(session_id)
        
        return {
            "success": True,
            "scenarios": forecast_state.get("scenarios", [])
        }
        
    except Exception as e:
        logger.error(f"Error getting scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints

@router.get("/sessions/{session_id}/locks")
async def get_edit_locks(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get current edit locks for the session"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Get budget state which includes edit locks
        budget_state = await collaborative_budget_planner.get_budget_state(session_id)
        
        if not budget_state:
            return {"success": True, "edit_locks": {}}
        
        return {
            "success": True,
            "edit_locks": budget_state.get("edit_locks", {}),
            "pending_conflicts": budget_state.get("pending_conflicts", [])
        }
        
    except Exception as e:
        logger.error(f"Error getting edit locks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}/locks/{item_path}")
async def release_edit_lock(
    session_id: str,
    item_path: str,
    current_user: User = Depends(get_current_user)
):
    """Release an edit lock (admin only or lock owner)"""
    try:
        # Verify user is participant in session
        participants = await session_manager.get_session_participants(session_id)
        participant_ids = [p["user_id"] for p in participants]
        
        if str(current_user.id) not in participant_ids:
            raise HTTPException(status_code=403, detail="Not a participant in this session")
        
        # Check if user owns the lock or is admin
        budget_state = await collaborative_budget_planner.get_budget_state(session_id)
        if budget_state:
            edit_locks = budget_state.get("edit_locks", {})
            lock_owner = edit_locks.get(item_path)
            
            if lock_owner and lock_owner != str(current_user.id):
                # Check if user is session moderator/owner
                user_participant = next(
                    (p for p in participants if p["user_id"] == str(current_user.id)), 
                    None
                )
                if not user_participant or user_participant.get("role") not in ["owner", "moderator"]:
                    raise HTTPException(status_code=403, detail="Cannot release lock owned by another user")
        
        # Release lock through budget planner
        if hasattr(collaborative_budget_planner, '_release_edit_lock'):
            await collaborative_budget_planner._release_edit_lock(session_id, item_path, str(current_user.id))
        
        return {"success": True, "message": "Lock released"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error releasing edit lock: {e}")
        raise HTTPException(status_code=500, detail=str(e))