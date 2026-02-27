"""
Collaborative Budget Planner for Real-Time FinOps Workspace

This module implements multi-user budget editing, real-time calculation updates,
conflict detection and resolution for budget constraints, and collaborative forecasting.
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .collaboration_models import (
    CollaborativeSession, SessionParticipant, ParticipantRole, ParticipantStatus
)
from .collaborative_session_manager import session_manager
from .state_synchronization import sync_engine, SyncEvent, SyncEventType
from .operational_transformation import Operation, OperationType, ot_engine
from .budget_management_system import Budget, BudgetPeriod, BudgetDimension
from .redis_config import redis_manager

logger = logging.getLogger(__name__)

class BudgetEditType(Enum):
    """Types of budget edits"""
    CATEGORY_ADD = "category_add"
    CATEGORY_REMOVE = "category_remove"
    CATEGORY_RENAME = "category_rename"
    LINE_ITEM_ADD = "line_item_add"
    LINE_ITEM_REMOVE = "line_item_remove"
    LINE_ITEM_UPDATE = "line_item_update"
    AMOUNT_CHANGE = "amount_change"
    ALLOCATION_CHANGE = "allocation_change"
    CONSTRAINT_ADD = "constraint_add"
    CONSTRAINT_REMOVE = "constraint_remove"
    CONSTRAINT_UPDATE = "constraint_update"

class ConflictType(Enum):
    """Types of budget conflicts"""
    CONCURRENT_EDIT = "concurrent_edit"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ALLOCATION_OVERFLOW = "allocation_overflow"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    PERMISSION_CONFLICT = "permission_conflict"

@dataclass
class BudgetLineItem:
    """Individual budget line item"""
    item_id: str
    category_id: str
    name: str
    description: str
    amount: Decimal
    allocated_amount: Decimal = Decimal('0')
    confidence_level: float = 1.0  # 0.0 to 1.0
    contributors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified_by: str = ""
    last_modified_at: datetime = field(default_factory=datetime.utcnow)
    is_locked: bool = False
    locked_by: Optional[str] = None
    locked_at: Optional[datetime] = None

@dataclass
class BudgetCategory:
    """Budget category containing line items"""
    category_id: str
    name: str
    description: str
    total_amount: Decimal
    allocated_amount: Decimal = Decimal('0')
    line_items: List[BudgetLineItem] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    color: str = "#007bff"
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified_by: str = ""
    last_modified_at: datetime = field(default_factory=datetime.utcnow)
    is_locked: bool = False
    locked_by: Optional[str] = None
    locked_at: Optional[datetime] = None

@dataclass
class BudgetConstraint:
    """Budget constraint definition"""
    constraint_id: str
    name: str
    constraint_type: str  # "max_amount", "min_amount", "percentage", "dependency"
    target_path: str  # Path to the budget element (category.item)
    value: Any
    operator: str  # "<=", ">=", "==", "!=", "<", ">"
    message: str
    severity: str = "error"  # "error", "warning", "info"
    is_active: bool = True
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BudgetConflict:
    """Budget editing conflict"""
    conflict_id: str
    conflict_type: ConflictType
    session_id: str
    participants: List[str]
    target_path: str
    conflicting_operations: List[Dict[str, Any]]
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CollaborativeBudget:
    """Collaborative budget with real-time editing capabilities"""
    budget_id: str
    session_id: str
    name: str
    description: str
    total_amount: Decimal
    currency: str = "USD"
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    categories: List[BudgetCategory] = field(default_factory=list)
    constraints: List[BudgetConstraint] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    version: int = 1
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified_by: str = ""
    last_modified_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def allocated_amount(self) -> Decimal:
        """Calculate total allocated amount across all categories"""
        return sum(category.allocated_amount for category in self.categories)
    
    @property
    def remaining_amount(self) -> Decimal:
        """Calculate remaining unallocated amount"""
        return self.total_amount - self.allocated_amount
    
    @property
    def allocation_percentage(self) -> float:
        """Calculate allocation percentage"""
        if self.total_amount == 0:
            return 0.0
        return float(self.allocated_amount / self.total_amount * 100)

class CollaborativeBudgetPlanner:
    """
    Multi-user budget editing system with real-time synchronization,
    conflict detection and resolution for budget constraints
    """
    
    def __init__(self):
        self.active_budgets: Dict[str, CollaborativeBudget] = {}  # session_id -> budget
        self.budget_locks: Dict[str, asyncio.Lock] = {}  # session_id -> lock
        self.edit_locks: Dict[str, Dict[str, str]] = {}  # session_id -> {path -> user_id}
        self.pending_conflicts: Dict[str, List[BudgetConflict]] = {}  # session_id -> conflicts
        
    async def create_budget_session(self, session_id: str, budget_config: Dict[str, Any], 
                                  participants: List[str]) -> CollaborativeBudget:
        """
        Create a new collaborative budget session
        
        Args:
            session_id: ID of the collaborative session
            budget_config: Budget configuration parameters
            participants: List of participant user IDs
            
        Returns:
            CollaborativeBudget: The created collaborative budget
        """
        try:
            # Create collaborative budget
            budget = CollaborativeBudget(
                budget_id=str(uuid.uuid4()),
                session_id=session_id,
                name=budget_config.get("name", "Collaborative Budget"),
                description=budget_config.get("description", ""),
                total_amount=Decimal(str(budget_config.get("total_amount", 0))),
                currency=budget_config.get("currency", "USD"),
                period=BudgetPeriod(budget_config.get("period", "monthly")),
                participants=participants.copy(),
                created_by=budget_config.get("created_by", "")
            )
            
            # Initialize with default categories if provided
            if "categories" in budget_config:
                for cat_config in budget_config["categories"]:
                    category = BudgetCategory(
                        category_id=str(uuid.uuid4()),
                        name=cat_config.get("name", ""),
                        description=cat_config.get("description", ""),
                        total_amount=Decimal(str(cat_config.get("amount", 0))),
                        created_by=budget.created_by
                    )
                    budget.categories.append(category)
            
            # Store budget and initialize locks
            self.active_budgets[session_id] = budget
            self.budget_locks[session_id] = asyncio.Lock()
            self.edit_locks[session_id] = {}
            self.pending_conflicts[session_id] = []
            
            # Cache budget in Redis for persistence
            await self._cache_budget(budget)
            
            # Broadcast budget creation event
            creation_event = SyncEvent(
                event_id=f"budget_created_{budget.budget_id}",
                event_type=SyncEventType.STATE_UPDATE,
                session_id=session_id,
                user_id=budget.created_by,
                data={
                    "budget": self._serialize_budget(budget),
                    "operation_type": "budget_created"
                },
                priority=3
            )
            
            await sync_engine.sync_state_update(
                session_id, 
                budget.created_by,
                Operation(
                    operation_type=OperationType.BUDGET_CREATE,
                    target_path="budget",
                    new_value=self._serialize_budget(budget),
                    user_id=budget.created_by,
                    session_id=session_id
                )
            )
            
            logger.info(f"Created collaborative budget {budget.name} for session {session_id}")
            return budget
            
        except Exception as e:
            logger.error(f"Error creating budget session: {e}")
            raise RuntimeError(f"Failed to create budget session: {e}")
    
    async def edit_budget_item(self, session_id: str, item_path: str, 
                             changes: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Edit a budget item with concurrent editing support and conflict resolution
        
        Args:
            session_id: ID of the collaborative session
            item_path: Path to the budget item (e.g., "categories.0.line_items.1")
            changes: Changes to apply
            user_id: ID of the user making the edit
            
        Returns:
            Dict containing edit result and any conflicts
        """
        try:
            if session_id not in self.active_budgets:
                raise ValueError(f"No active budget for session {session_id}")
            
            # Acquire budget lock
            async with self.budget_locks[session_id]:
                budget = self.active_budgets[session_id]
                
                # Check for edit conflicts
                conflict_result = await self._check_edit_conflicts(
                    session_id, item_path, changes, user_id
                )
                
                if conflict_result["has_conflicts"]:
                    return {
                        "success": False,
                        "conflicts": conflict_result["conflicts"],
                        "requires_resolution": True
                    }
                
                # Acquire edit lock for this path
                if not await self._acquire_edit_lock(session_id, item_path, user_id):
                    return {
                        "success": False,
                        "error": f"Item is currently being edited by another user",
                        "locked_by": self.edit_locks[session_id].get(item_path)
                    }
                
                try:
                    # Apply changes
                    old_value = self._get_item_by_path(budget, item_path)
                    new_value = await self._apply_changes(budget, item_path, changes, user_id)
                    
                    # Validate constraints
                    constraint_violations = await self._validate_constraints(budget, item_path, new_value)
                    if constraint_violations:
                        # Rollback changes
                        self._set_item_by_path(budget, item_path, old_value)
                        return {
                            "success": False,
                            "constraint_violations": constraint_violations
                        }
                    
                    # Update budget version and timestamps
                    budget.version += 1
                    budget.last_modified_by = user_id
                    budget.last_modified_at = datetime.utcnow()
                    
                    # Recalculate totals
                    await self._recalculate_totals(budget)
                    
                    # Cache updated budget
                    await self._cache_budget(budget)
                    
                    # Create operation for synchronization
                    operation = Operation(
                        operation_type=OperationType.BUDGET_EDIT,
                        target_path=item_path,
                        old_value=self._serialize_value(old_value),
                        new_value=self._serialize_value(new_value),
                        user_id=user_id,
                        session_id=session_id,
                        metadata={
                            "edit_type": changes.get("edit_type", "update"),
                            "budget_version": budget.version
                        }
                    )
                    
                    # Synchronize changes
                    await sync_engine.sync_state_update(session_id, user_id, operation)
                    
                    # Broadcast real-time calculation updates
                    await self._broadcast_calculation_updates(session_id, budget, user_id)
                    
                    logger.info(f"User {user_id} edited budget item {item_path} in session {session_id}")
                    
                    return {
                        "success": True,
                        "new_value": self._serialize_value(new_value),
                        "budget_totals": {
                            "total_amount": str(budget.total_amount),
                            "allocated_amount": str(budget.allocated_amount),
                            "remaining_amount": str(budget.remaining_amount),
                            "allocation_percentage": budget.allocation_percentage
                        },
                        "version": budget.version
                    }
                    
                finally:
                    # Release edit lock
                    await self._release_edit_lock(session_id, item_path, user_id)
                    
        except Exception as e:
            logger.error(f"Error editing budget item: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_budget_category(self, session_id: str, category_config: Dict[str, Any], 
                                user_id: str) -> Dict[str, Any]:
        """Add a new budget category with real-time updates"""
        try:
            if session_id not in self.active_budgets:
                raise ValueError(f"No active budget for session {session_id}")
            
            async with self.budget_locks[session_id]:
                budget = self.active_budgets[session_id]
                
                # Create new category
                category = BudgetCategory(
                    category_id=str(uuid.uuid4()),
                    name=category_config.get("name", "New Category"),
                    description=category_config.get("description", ""),
                    total_amount=Decimal(str(category_config.get("amount", 0))),
                    created_by=user_id
                )
                
                # Add to budget
                budget.categories.append(category)
                budget.version += 1
                budget.last_modified_by = user_id
                budget.last_modified_at = datetime.utcnow()
                
                # Recalculate totals
                await self._recalculate_totals(budget)
                
                # Cache updated budget
                await self._cache_budget(budget)
                
                # Synchronize changes
                operation = Operation(
                    operation_type=OperationType.BUDGET_EDIT,
                    target_path=f"categories.{len(budget.categories)-1}",
                    new_value=self._serialize_category(category),
                    user_id=user_id,
                    session_id=session_id,
                    metadata={"edit_type": "category_add"}
                )
                
                await sync_engine.sync_state_update(session_id, user_id, operation)
                await self._broadcast_calculation_updates(session_id, budget, user_id)
                
                return {
                    "success": True,
                    "category": self._serialize_category(category),
                    "budget_totals": {
                        "total_amount": str(budget.total_amount),
                        "allocated_amount": str(budget.allocated_amount),
                        "remaining_amount": str(budget.remaining_amount)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error adding budget category: {e}")
            return {"success": False, "error": str(e)}
    
    async def add_line_item(self, session_id: str, category_id: str, 
                          item_config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Add a new line item to a budget category"""
        try:
            if session_id not in self.active_budgets:
                raise ValueError(f"No active budget for session {session_id}")
            
            async with self.budget_locks[session_id]:
                budget = self.active_budgets[session_id]
                
                # Find category
                category = None
                category_index = None
                for i, cat in enumerate(budget.categories):
                    if cat.category_id == category_id:
                        category = cat
                        category_index = i
                        break
                
                if not category:
                    return {"success": False, "error": "Category not found"}
                
                # Create new line item
                line_item = BudgetLineItem(
                    item_id=str(uuid.uuid4()),
                    category_id=category_id,
                    name=item_config.get("name", "New Item"),
                    description=item_config.get("description", ""),
                    amount=Decimal(str(item_config.get("amount", 0))),
                    confidence_level=item_config.get("confidence_level", 1.0),
                    created_by=user_id
                )
                
                # Add to category
                category.line_items.append(line_item)
                budget.version += 1
                budget.last_modified_by = user_id
                budget.last_modified_at = datetime.utcnow()
                
                # Recalculate totals
                await self._recalculate_totals(budget)
                
                # Cache updated budget
                await self._cache_budget(budget)
                
                # Synchronize changes
                item_path = f"categories.{category_index}.line_items.{len(category.line_items)-1}"
                operation = Operation(
                    operation_type=OperationType.BUDGET_EDIT,
                    target_path=item_path,
                    new_value=self._serialize_line_item(line_item),
                    user_id=user_id,
                    session_id=session_id,
                    metadata={"edit_type": "line_item_add"}
                )
                
                await sync_engine.sync_state_update(session_id, user_id, operation)
                await self._broadcast_calculation_updates(session_id, budget, user_id)
                
                return {
                    "success": True,
                    "line_item": self._serialize_line_item(line_item),
                    "category_totals": {
                        "total_amount": str(category.total_amount),
                        "allocated_amount": str(category.allocated_amount)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error adding line item: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_budget_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current collaborative budget state"""
        try:
            if session_id not in self.active_budgets:
                # Try to load from cache
                budget = await self._load_budget_from_cache(session_id)
                if budget:
                    self.active_budgets[session_id] = budget
                    self.budget_locks[session_id] = asyncio.Lock()
                    self.edit_locks[session_id] = {}
                    self.pending_conflicts[session_id] = []
                else:
                    return None
            
            budget = self.active_budgets[session_id]
            
            return {
                "budget": self._serialize_budget(budget),
                "edit_locks": self.edit_locks[session_id].copy(),
                "pending_conflicts": [
                    self._serialize_conflict(conflict) 
                    for conflict in self.pending_conflicts[session_id]
                ],
                "totals": {
                    "total_amount": str(budget.total_amount),
                    "allocated_amount": str(budget.allocated_amount),
                    "remaining_amount": str(budget.remaining_amount),
                    "allocation_percentage": budget.allocation_percentage
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting budget state: {e}")
            return None
    
    # Private helper methods
    
    async def _check_edit_conflicts(self, session_id: str, item_path: str, 
                                  changes: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Check for editing conflicts"""
        conflicts = []
        
        # Check if item is locked by another user
        if item_path in self.edit_locks[session_id]:
            locked_by = self.edit_locks[session_id][item_path]
            if locked_by != user_id:
                conflicts.append({
                    "type": ConflictType.CONCURRENT_EDIT.value,
                    "message": f"Item is being edited by another user",
                    "locked_by": locked_by
                })
        
        # Check for constraint violations (preliminary)
        budget = self.active_budgets[session_id]
        if "amount" in changes:
            # Check if amount change would violate constraints
            for constraint in budget.constraints:
                if constraint.target_path == item_path and constraint.is_active:
                    if not self._evaluate_constraint(constraint, changes["amount"]):
                        conflicts.append({
                            "type": ConflictType.CONSTRAINT_VIOLATION.value,
                            "constraint": constraint.name,
                            "message": constraint.message
                        })
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts
        }
    
    async def _acquire_edit_lock(self, session_id: str, item_path: str, user_id: str) -> bool:
        """Acquire edit lock for an item"""
        if item_path in self.edit_locks[session_id]:
            return self.edit_locks[session_id][item_path] == user_id
        
        self.edit_locks[session_id][item_path] = user_id
        
        # Set lock timeout (5 minutes)
        asyncio.create_task(self._auto_release_lock(session_id, item_path, user_id, 300))
        
        return True
    
    async def _release_edit_lock(self, session_id: str, item_path: str, user_id: str):
        """Release edit lock for an item"""
        if (item_path in self.edit_locks[session_id] and 
            self.edit_locks[session_id][item_path] == user_id):
            del self.edit_locks[session_id][item_path]
    
    async def _auto_release_lock(self, session_id: str, item_path: str, user_id: str, timeout: int):
        """Automatically release lock after timeout"""
        await asyncio.sleep(timeout)
        await self._release_edit_lock(session_id, item_path, user_id)
    
    def _get_item_by_path(self, budget: CollaborativeBudget, path: str) -> Any:
        """Get budget item by path"""
        parts = path.split('.')
        current = budget
        
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        
        return current
    
    def _set_item_by_path(self, budget: CollaborativeBudget, path: str, value: Any):
        """Set budget item by path"""
        parts = path.split('.')
        current = budget
        
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        
        final_part = parts[-1]
        if final_part.isdigit():
            current[int(final_part)] = value
        else:
            setattr(current, final_part, value)
    
    async def _apply_changes(self, budget: CollaborativeBudget, path: str, 
                           changes: Dict[str, Any], user_id: str) -> Any:
        """Apply changes to budget item"""
        item = self._get_item_by_path(budget, path)
        
        # Apply changes based on item type
        if isinstance(item, BudgetLineItem):
            if "name" in changes:
                item.name = changes["name"]
            if "description" in changes:
                item.description = changes["description"]
            if "amount" in changes:
                item.amount = Decimal(str(changes["amount"]))
            if "confidence_level" in changes:
                item.confidence_level = float(changes["confidence_level"])
            
            item.last_modified_by = user_id
            item.last_modified_at = datetime.utcnow()
            
        elif isinstance(item, BudgetCategory):
            if "name" in changes:
                item.name = changes["name"]
            if "description" in changes:
                item.description = changes["description"]
            if "total_amount" in changes:
                item.total_amount = Decimal(str(changes["total_amount"]))
            
            item.last_modified_by = user_id
            item.last_modified_at = datetime.utcnow()
        
        return item
    
    async def _validate_constraints(self, budget: CollaborativeBudget, 
                                  path: str, value: Any) -> List[Dict[str, Any]]:
        """Validate budget constraints"""
        violations = []
        
        for constraint in budget.constraints:
            if constraint.target_path == path and constraint.is_active:
                if not self._evaluate_constraint(constraint, value):
                    violations.append({
                        "constraint_id": constraint.constraint_id,
                        "name": constraint.name,
                        "message": constraint.message,
                        "severity": constraint.severity
                    })
        
        return violations
    
    def _evaluate_constraint(self, constraint: BudgetConstraint, value: Any) -> bool:
        """Evaluate a single constraint"""
        try:
            constraint_value = constraint.value
            
            if constraint.operator == "<=":
                return value <= constraint_value
            elif constraint.operator == ">=":
                return value >= constraint_value
            elif constraint.operator == "==":
                return value == constraint_value
            elif constraint.operator == "!=":
                return value != constraint_value
            elif constraint.operator == "<":
                return value < constraint_value
            elif constraint.operator == ">":
                return value > constraint_value
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating constraint: {e}")
            return False
    
    async def _recalculate_totals(self, budget: CollaborativeBudget):
        """Recalculate budget totals and allocations"""
        for category in budget.categories:
            # Recalculate category allocated amount
            category.allocated_amount = sum(
                item.amount for item in category.line_items
            )
    
    async def _broadcast_calculation_updates(self, session_id: str, 
                                          budget: CollaborativeBudget, user_id: str):
        """Broadcast real-time calculation updates to all participants"""
        try:
            update_event = SyncEvent(
                event_id=f"calc_update_{int(datetime.utcnow().timestamp() * 1000)}",
                event_type=SyncEventType.STATE_UPDATE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "operation_type": "calculation_update",
                    "totals": {
                        "total_amount": str(budget.total_amount),
                        "allocated_amount": str(budget.allocated_amount),
                        "remaining_amount": str(budget.remaining_amount),
                        "allocation_percentage": budget.allocation_percentage
                    },
                    "category_totals": [
                        {
                            "category_id": cat.category_id,
                            "total_amount": str(cat.total_amount),
                            "allocated_amount": str(cat.allocated_amount)
                        }
                        for cat in budget.categories
                    ],
                    "version": budget.version
                },
                priority=2
            )
            
            # Queue for immediate broadcast
            await sync_engine._queue_sync_event(session_id, update_event)
            
        except Exception as e:
            logger.error(f"Error broadcasting calculation updates: {e}")
    
    async def _cache_budget(self, budget: CollaborativeBudget):
        """Cache budget in Redis"""
        try:
            cache_key = f"collaborative_budget:{budget.session_id}"
            budget_data = self._serialize_budget(budget)
            
            # Use standard Redis client methods
            client = await redis_manager.get_async_client()
            import json
            await client.set(cache_key, json.dumps(budget_data), ex=3600)  # 1 hour TTL
        except Exception as e:
            logger.error(f"Error caching budget: {e}")
    
    async def _load_budget_from_cache(self, session_id: str) -> Optional[CollaborativeBudget]:
        """Load budget from Redis cache"""
        try:
            cache_key = f"collaborative_budget:{session_id}"
            client = await redis_manager.get_async_client()
            import json
            
            budget_json = await client.get(cache_key)
            if budget_json:
                budget_data = json.loads(budget_json)
                return self._deserialize_budget(budget_data)
            return None
        except Exception as e:
            logger.error(f"Error loading budget from cache: {e}")
            return None
    
    def _serialize_budget(self, budget: CollaborativeBudget) -> Dict[str, Any]:
        """Serialize budget to dictionary"""
        return {
            "budget_id": budget.budget_id,
            "session_id": budget.session_id,
            "name": budget.name,
            "description": budget.description,
            "total_amount": str(budget.total_amount),
            "currency": budget.currency,
            "period": budget.period.value,
            "categories": [self._serialize_category(cat) for cat in budget.categories],
            "constraints": [self._serialize_constraint(const) for const in budget.constraints],
            "participants": budget.participants,
            "version": budget.version,
            "created_by": budget.created_by,
            "created_at": budget.created_at.isoformat(),
            "last_modified_by": budget.last_modified_by,
            "last_modified_at": budget.last_modified_at.isoformat()
        }
    
    def _serialize_category(self, category: BudgetCategory) -> Dict[str, Any]:
        """Serialize budget category to dictionary"""
        return {
            "category_id": category.category_id,
            "name": category.name,
            "description": category.description,
            "total_amount": str(category.total_amount),
            "allocated_amount": str(category.allocated_amount),
            "line_items": [self._serialize_line_item(item) for item in category.line_items],
            "constraints": category.constraints,
            "color": category.color,
            "created_by": category.created_by,
            "created_at": category.created_at.isoformat(),
            "last_modified_by": category.last_modified_by,
            "last_modified_at": category.last_modified_at.isoformat(),
            "is_locked": category.is_locked,
            "locked_by": category.locked_by,
            "locked_at": category.locked_at.isoformat() if category.locked_at else None
        }
    
    def _serialize_line_item(self, item: BudgetLineItem) -> Dict[str, Any]:
        """Serialize budget line item to dictionary"""
        return {
            "item_id": item.item_id,
            "category_id": item.category_id,
            "name": item.name,
            "description": item.description,
            "amount": str(item.amount),
            "allocated_amount": str(item.allocated_amount),
            "confidence_level": item.confidence_level,
            "contributors": item.contributors,
            "tags": item.tags,
            "metadata": item.metadata,
            "created_by": item.created_by,
            "created_at": item.created_at.isoformat(),
            "last_modified_by": item.last_modified_by,
            "last_modified_at": item.last_modified_at.isoformat(),
            "is_locked": item.is_locked,
            "locked_by": item.locked_by,
            "locked_at": item.locked_at.isoformat() if item.locked_at else None
        }
    
    def _serialize_constraint(self, constraint: BudgetConstraint) -> Dict[str, Any]:
        """Serialize budget constraint to dictionary"""
        return {
            "constraint_id": constraint.constraint_id,
            "name": constraint.name,
            "constraint_type": constraint.constraint_type,
            "target_path": constraint.target_path,
            "value": constraint.value,
            "operator": constraint.operator,
            "message": constraint.message,
            "severity": constraint.severity,
            "is_active": constraint.is_active,
            "created_by": constraint.created_by,
            "created_at": constraint.created_at.isoformat()
        }
    
    def _serialize_conflict(self, conflict: BudgetConflict) -> Dict[str, Any]:
        """Serialize budget conflict to dictionary"""
        return {
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type.value,
            "session_id": conflict.session_id,
            "participants": conflict.participants,
            "target_path": conflict.target_path,
            "conflicting_operations": conflict.conflicting_operations,
            "resolution_strategy": conflict.resolution_strategy,
            "resolved": conflict.resolved,
            "resolved_by": conflict.resolved_by,
            "resolved_at": conflict.resolved_at.isoformat() if conflict.resolved_at else None,
            "created_at": conflict.created_at.isoformat()
        }
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize any value for JSON storage"""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif hasattr(value, '__dict__'):
            return value.__dict__
        else:
            return value
    
    def _deserialize_budget(self, data: Dict[str, Any]) -> CollaborativeBudget:
        """Deserialize budget from dictionary"""
        # This is a simplified deserialization - in production would be more robust
        budget = CollaborativeBudget(
            budget_id=data["budget_id"],
            session_id=data["session_id"],
            name=data["name"],
            description=data["description"],
            total_amount=Decimal(data["total_amount"]),
            currency=data["currency"],
            period=BudgetPeriod(data["period"]),
            participants=data["participants"],
            version=data["version"],
            created_by=data["created_by"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_modified_by=data["last_modified_by"],
            last_modified_at=datetime.fromisoformat(data["last_modified_at"])
        )
        
        # Deserialize categories (simplified)
        for cat_data in data["categories"]:
            category = BudgetCategory(
                category_id=cat_data["category_id"],
                name=cat_data["name"],
                description=cat_data["description"],
                total_amount=Decimal(cat_data["total_amount"]),
                allocated_amount=Decimal(cat_data["allocated_amount"]),
                created_by=cat_data["created_by"],
                created_at=datetime.fromisoformat(cat_data["created_at"])
            )
            budget.categories.append(category)
        
        return budget

# Global collaborative budget planner instance
collaborative_budget_planner = CollaborativeBudgetPlanner()