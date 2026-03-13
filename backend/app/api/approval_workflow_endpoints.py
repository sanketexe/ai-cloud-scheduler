"""
API Endpoints for Approval Workflow Engine

This module provides REST API endpoints for the approval workflow engine,
enabling integration with the collaborative FinOps workspace frontend.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from app.database.database import get_db_session
from app.services.startup_migration.models import User
from app.services.approval_workflow_engine import (
    workflow_engine, OptimizationAction, DecisionContext, 
    ApprovalDecision, Priority, EscalationReason
)
from app.services.decision_tracking_system import decision_tracker, notification_service, NotificationType
from app.core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/approval-workflow", tags=["Approval Workflow"])

# Request/Response Models

class OptimizationActionRequest(BaseModel):
    """Request model for optimization actions"""
    action_id: str
    action_type: str
    resource_id: str
    resource_type: str
    cost_impact: float = Field(..., ge=0)
    risk_level: str = Field(..., pattern="^(low|medium|high)$")
    description: str
    proposed_changes: Dict[str, Any] = {}
    estimated_savings: float = Field(..., ge=0)
    implementation_timeline: str

class DecisionContextRequest(BaseModel):
    """Request model for decision context"""
    session_id: str
    participants: List[str]
    discussion_summary: str
    data_analyzed: List[D
