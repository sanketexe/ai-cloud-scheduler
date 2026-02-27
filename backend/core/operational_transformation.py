"""
Operational Transformation System for Real-Time Collaborative FinOps Workspace

This module implements operational transformation algorithms for conflict-free concurrent editing
in collaborative sessions, ensuring consistent state across all participants.
"""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.exc import IntegrityError

from .database import get_db_session
from .collaboration_models import SessionStateUpdate, CollaborativeSession
from .redis_config import redis_manager

logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Types of operations that can be performed"""
    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"
    REPLACE = "replace"
    FILTER_CHANGE = "filter_change"
    VIEW_CHANGE = "view_change"
    DASHBOARD_CONFIG = "dashboard_config"
    ANNOTATION_ADD = "annotation_add"
    ANNOTATION_REMOVE = "annotation_remove"
    CURSOR_MOVE = "cursor_move"
    BUDGET_CREATE = "budget_create"
    BUDGET_EDIT = "budget_edit"
    BUDGET_DELETE = "budget_delete"

class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts"""
    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MERGE = "merge"
    USER_CHOICE = "user_choice"
    OPERATIONAL_TRANSFORM = "operational_transform"

@dataclass
class Operation:
    """Represents a single operation in the collaborative system"""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType = OperationType.RETAIN
    target_path: str = ""
    position: int = 0
    length: int = 0
    content: Any = None
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""
    session_id: str = ""
    version: int = 0
    parent_version: Optional[int] = None
    
    def __post_init__(self):
        """Validate operation after initialization"""
        if not self.operation_id:
            self.operation_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()

@dataclass
class TransformResult:
    """Result of operation transformation"""
    transformed_operation: Operation
    conflicts_detected: List[str] = field(default_factory=list)
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPERATIONAL_TRANSFORM
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateVersion:
    """Represents a version of the collaborative state"""
    version: int
    operations: List[Operation] = field(default_factory=list)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

@dataclass
class ConflictInfo:
    """Information about a detected conflict"""
    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_a: Operation = None
    operation_b: Operation = None
    conflict_type: str = ""
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPERATIONAL_TRANSFORM
    resolved: bool = False
    resolution_data: Dict[str, Any] = field(default_factory=dict)

class OperationTransformer(ABC):
    """Abstract base class for operation transformers"""
    
    @abstractmethod
    def transform(self, op_a: Operation, op_b: Operation) -> Tuple[Operation, Operation]:
        """Transform two concurrent operations"""
        pass
    
    @abstractmethod
    def can_transform(self, op_a: Operation, op_b: Operation) -> bool:
        """Check if this transformer can handle the given operations"""
        pass

class TextOperationTransformer(OperationTransformer):
    """Transformer for text-based operations (INSERT, DELETE, RETAIN)"""
    
    def can_transform(self, op_a: Operation, op_b: Operation) -> bool:
        """Check if both operations are text operations on the same path"""
        text_ops = {OperationType.INSERT, OperationType.DELETE, OperationType.RETAIN}
        return (op_a.operation_type in text_ops and 
                op_b.operation_type in text_ops and
                op_a.target_path == op_b.target_path)
    
    def transform(self, op_a: Operation, op_b: Operation) -> Tuple[Operation, Operation]:
        """Transform two text operations using standard OT algorithms"""
        try:
            # Create copies to avoid modifying originals
            transformed_a = Operation(**op_a.__dict__)
            transformed_b = Operation(**op_b.__dict__)
            
            # Handle INSERT vs INSERT
            if (op_a.operation_type == OperationType.INSERT and 
                op_b.operation_type == OperationType.INSERT):
                
                if op_a.position <= op_b.position:
                    # op_a comes before op_b, adjust op_b position
                    transformed_b.position += op_a.length
                else:
                    # op_b comes before op_a, adjust op_a position
                    transformed_a.position += op_b.length
            
            # Handle INSERT vs DELETE
            elif (op_a.operation_type == OperationType.INSERT and 
                  op_b.operation_type == OperationType.DELETE):
                
                if op_a.position <= op_b.position:
                    # Insert before delete, adjust delete position
                    transformed_b.position += op_a.length
                else:
                    # Insert after delete start, adjust insert position
                    if op_a.position <= op_b.position + op_b.length:
                        # Insert within deleted range, move to delete start
                        transformed_a.position = op_b.position
                    else:
                        # Insert after deleted range
                        transformed_a.position -= op_b.length
            
            # Handle DELETE vs INSERT (symmetric to INSERT vs DELETE)
            elif (op_a.operation_type == OperationType.DELETE and 
                  op_b.operation_type == OperationType.INSERT):
                
                if op_b.position <= op_a.position:
                    # Insert before delete, adjust delete position
                    transformed_a.position += op_b.length
                else:
                    # Insert after delete start
                    if op_b.position <= op_a.position + op_a.length:
                        # Insert within deleted range, move to delete start
                        transformed_b.position = op_a.position
                    else:
                        # Insert after deleted range
                        transformed_b.position -= op_a.length
            
            # Handle DELETE vs DELETE
            elif (op_a.operation_type == OperationType.DELETE and 
                  op_b.operation_type == OperationType.DELETE):
                
                # Calculate overlap
                a_end = op_a.position + op_a.length
                b_end = op_b.position + op_b.length
                
                if op_a.position >= b_end:
                    # op_a after op_b, adjust op_a position
                    transformed_a.position -= op_b.length
                elif op_b.position >= a_end:
                    # op_b after op_a, adjust op_b position
                    transformed_b.position -= op_a.length
                else:
                    # Overlapping deletes - complex case
                    overlap_start = max(op_a.position, op_b.position)
                    overlap_end = min(a_end, b_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    
                    if op_a.position <= op_b.position:
                        # op_a starts first
                        transformed_a.length -= overlap_length
                        transformed_b.position = op_a.position
                        transformed_b.length -= overlap_length
                    else:
                        # op_b starts first
                        transformed_b.length -= overlap_length
                        transformed_a.position = op_b.position
                        transformed_a.length -= overlap_length
            
            return transformed_a, transformed_b
            
        except Exception as e:
            logger.error(f"Error transforming text operations: {e}")
            # Return original operations if transformation fails
            return op_a, op_b

class FilterOperationTransformer(OperationTransformer):
    """Transformer for filter change operations"""
    
    def can_transform(self, op_a: Operation, op_b: Operation) -> bool:
        """Check if both operations are filter operations"""
        return (op_a.operation_type == OperationType.FILTER_CHANGE and
                op_b.operation_type == OperationType.FILTER_CHANGE)
    
    def transform(self, op_a: Operation, op_b: Operation) -> Tuple[Operation, Operation]:
        """Transform filter operations - typically merge or last-writer-wins"""
        try:
            transformed_a = Operation(**op_a.__dict__)
            transformed_b = Operation(**op_b.__dict__)
            
            # If same filter path, use timestamp to determine precedence
            if op_a.target_path == op_b.target_path:
                if op_a.timestamp <= op_b.timestamp:
                    # op_b wins, mark op_a as superseded
                    transformed_a.metadata["superseded"] = True
                    transformed_a.metadata["superseded_by"] = op_b.operation_id
                else:
                    # op_a wins, mark op_b as superseded
                    transformed_b.metadata["superseded"] = True
                    transformed_b.metadata["superseded_by"] = op_a.operation_id
            
            # Different filter paths can coexist
            return transformed_a, transformed_b
            
        except Exception as e:
            logger.error(f"Error transforming filter operations: {e}")
            return op_a, op_b

class DashboardConfigTransformer(OperationTransformer):
    """Transformer for dashboard configuration operations"""
    
    def can_transform(self, op_a: Operation, op_b: Operation) -> bool:
        """Check if both operations are dashboard config operations"""
        return (op_a.operation_type == OperationType.DASHBOARD_CONFIG and
                op_b.operation_type == OperationType.DASHBOARD_CONFIG)
    
    def transform(self, op_a: Operation, op_b: Operation) -> Tuple[Operation, Operation]:
        """Transform dashboard config operations with intelligent merging"""
        try:
            transformed_a = Operation(**op_a.__dict__)
            transformed_b = Operation(**op_b.__dict__)
            
            # Merge configurations if they don't conflict
            if isinstance(op_a.new_value, dict) and isinstance(op_b.new_value, dict):
                a_keys = set(op_a.new_value.keys())
                b_keys = set(op_b.new_value.keys())
                
                # Check for conflicting keys
                conflicting_keys = a_keys.intersection(b_keys)
                
                if conflicting_keys:
                    # Use timestamp for conflict resolution
                    if op_a.timestamp <= op_b.timestamp:
                        # op_b wins for conflicting keys
                        merged_config = {**op_a.new_value, **op_b.new_value}
                        transformed_b.new_value = merged_config
                        transformed_a.metadata["merged_into"] = op_b.operation_id
                    else:
                        # op_a wins for conflicting keys
                        merged_config = {**op_b.new_value, **op_a.new_value}
                        transformed_a.new_value = merged_config
                        transformed_b.metadata["merged_into"] = op_a.operation_id
                else:
                    # No conflicts, merge both
                    merged_config = {**op_a.new_value, **op_b.new_value}
                    transformed_a.new_value = merged_config
                    transformed_b.new_value = merged_config
            
            return transformed_a, transformed_b
            
        except Exception as e:
            logger.error(f"Error transforming dashboard config operations: {e}")
            return op_a, op_b

class OperationalTransformationEngine:
    """
    Main engine for operational transformation in collaborative sessions
    """
    
    def __init__(self):
        self.transformers: List[OperationTransformer] = [
            TextOperationTransformer(),
            FilterOperationTransformer(),
            DashboardConfigTransformer()
        ]
        self.state_versions: Dict[str, List[StateVersion]] = {}  # session_id -> versions
        self.operation_locks: Dict[str, asyncio.Lock] = {}  # session_id -> lock
        
    async def apply_operation(self, session_id: str, operation: Operation) -> TransformResult:
        """
        Apply an operation to the session state with conflict resolution
        
        Args:
            session_id: ID of the collaborative session
            operation: Operation to apply
            
        Returns:
            TransformResult: Result of applying the operation
        """
        try:
            # Get or create session lock
            if session_id not in self.operation_locks:
                self.operation_locks[session_id] = asyncio.Lock()
            
            async with self.operation_locks[session_id]:
                # Get current state version
                current_version = await self._get_current_version(session_id)
                
                # Get concurrent operations since parent version
                concurrent_ops = await self._get_concurrent_operations(
                    session_id, operation.parent_version or current_version
                )
                
                # Transform operation against concurrent operations
                transformed_op = operation
                conflicts = []
                
                for concurrent_op in concurrent_ops:
                    if concurrent_op.operation_id != operation.operation_id:
                        transform_result = await self._transform_operations(
                            transformed_op, concurrent_op
                        )
                        transformed_op = transform_result.transformed_operation
                        conflicts.extend(transform_result.conflicts_detected)
                
                # Set final version
                transformed_op.version = current_version + 1
                
                # Store operation in database
                await self._store_operation(session_id, transformed_op)
                
                # Update state cache
                await self._update_state_cache(session_id, transformed_op)
                
                # Create new state version
                await self._create_state_version(session_id, transformed_op)
                
                return TransformResult(
                    transformed_operation=transformed_op,
                    conflicts_detected=conflicts,
                    resolution_strategy=ConflictResolutionStrategy.OPERATIONAL_TRANSFORM
                )
                
        except Exception as e:
            logger.error(f"Error applying operation: {e}")
            raise RuntimeError(f"Failed to apply operation: {e}")
    
    async def resolve_conflict(self, session_id: str, conflict: ConflictInfo) -> ConflictInfo:
        """
        Resolve a detected conflict between operations
        
        Args:
            session_id: ID of the collaborative session
            conflict: Conflict information
            
        Returns:
            ConflictInfo: Updated conflict with resolution
        """
        try:
            if conflict.resolution_strategy == ConflictResolutionStrategy.OPERATIONAL_TRANSFORM:
                # Use OT to resolve conflict
                transform_result = await self._transform_operations(
                    conflict.operation_a, conflict.operation_b
                )
                
                conflict.resolved = True
                conflict.resolution_data = {
                    "transformed_a": transform_result.transformed_operation.__dict__,
                    "conflicts": transform_result.conflicts_detected,
                    "resolution_method": "operational_transform"
                }
                
            elif conflict.resolution_strategy == ConflictResolutionStrategy.LAST_WRITER_WINS:
                # Use timestamp to determine winner
                if conflict.operation_a.timestamp <= conflict.operation_b.timestamp:
                    winner = conflict.operation_b
                    loser = conflict.operation_a
                else:
                    winner = conflict.operation_a
                    loser = conflict.operation_b
                
                conflict.resolved = True
                conflict.resolution_data = {
                    "winner": winner.__dict__,
                    "loser": loser.__dict__,
                    "resolution_method": "last_writer_wins"
                }
            
            # Store conflict resolution
            await self._store_conflict_resolution(session_id, conflict)
            
            return conflict
            
        except Exception as e:
            logger.error(f"Error resolving conflict: {e}")
            conflict.resolved = False
            conflict.resolution_data["error"] = str(e)
            return conflict
    
    async def get_current_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current state of a collaborative session
        
        Args:
            session_id: ID of the collaborative session
            
        Returns:
            Dict containing the current session state
        """
        try:
            # Try to get from cache first
            cache_key = f"session:{session_id}:state"
            cached_state = await redis_manager.get_json(cache_key)
            
            if cached_state:
                return cached_state
            
            # Rebuild state from operations if not cached
            return await self._rebuild_state_from_operations(session_id)
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {}
    
    async def rollback_to_version(self, session_id: str, target_version: int) -> bool:
        """
        Rollback session state to a specific version
        
        Args:
            session_id: ID of the collaborative session
            target_version: Version to rollback to
            
        Returns:
            bool: True if rollback was successful
        """
        try:
            async with get_db_session() as db:
                # Get all operations up to target version
                operations = db.query(SessionStateUpdate).filter(
                    SessionStateUpdate.session_id == session_id,
                    SessionStateUpdate.version <= target_version
                ).order_by(SessionStateUpdate.version).all()
                
                # Rebuild state from operations
                state = await self._apply_operations_to_state({}, operations)
                
                # Update cache with rollback state
                cache_key = f"session:{session_id}:state"
                await redis_manager.set_json(cache_key, state)
                
                # Mark later operations as rolled back
                db.query(SessionStateUpdate).filter(
                    SessionStateUpdate.session_id == session_id,
                    SessionStateUpdate.version > target_version
                ).update({"conflict_resolved": True})
                db.commit()
                
                logger.info(f"Rolled back session {session_id} to version {target_version}")
                return True
                
        except Exception as e:
            logger.error(f"Error rolling back to version: {e}")
            return False
    
    async def _transform_operations(self, op_a: Operation, op_b: Operation) -> TransformResult:
        """Transform two operations against each other"""
        try:
            # Find appropriate transformer
            for transformer in self.transformers:
                if transformer.can_transform(op_a, op_b):
                    transformed_a, transformed_b = transformer.transform(op_a, op_b)
                    
                    # Detect conflicts
                    conflicts = []
                    if hasattr(transformed_a, 'metadata') and transformed_a.metadata.get('superseded'):
                        conflicts.append(f"Operation {op_a.operation_id} superseded by {op_b.operation_id}")
                    
                    return TransformResult(
                        transformed_operation=transformed_a,
                        conflicts_detected=conflicts
                    )
            
            # No specific transformer found, use default behavior
            return TransformResult(
                transformed_operation=op_a,
                conflicts_detected=[f"No transformer found for operations {op_a.operation_type} and {op_b.operation_type}"]
            )
            
        except Exception as e:
            logger.error(f"Error transforming operations: {e}")
            return TransformResult(
                transformed_operation=op_a,
                conflicts_detected=[f"Transformation error: {e}"]
            )
    
    async def _get_current_version(self, session_id: str) -> int:
        """Get the current version number for a session"""
        try:
            async with get_db_session() as db:
                latest_update = db.query(SessionStateUpdate).filter(
                    SessionStateUpdate.session_id == session_id
                ).order_by(desc(SessionStateUpdate.version)).first()
                
                return latest_update.version if latest_update else 0
        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return 0
    
    async def _get_concurrent_operations(self, session_id: str, since_version: int) -> List[Operation]:
        """Get operations that occurred since a specific version"""
        try:
            async with get_db_session() as db:
                updates = db.query(SessionStateUpdate).filter(
                    SessionStateUpdate.session_id == session_id,
                    SessionStateUpdate.version > since_version
                ).order_by(SessionStateUpdate.version).all()
                
                operations = []
                for update in updates:
                    op = Operation(
                        operation_id=update.operation_id,
                        operation_type=OperationType(update.operation_type),
                        target_path=update.target_path,
                        old_value=update.old_value,
                        new_value=update.new_value,
                        user_id=str(update.user_id),
                        session_id=str(update.session_id),
                        version=update.version,
                        timestamp=update.created_at
                    )
                    operations.append(op)
                
                return operations
        except Exception as e:
            logger.error(f"Error getting concurrent operations: {e}")
            return []
    
    async def _store_operation(self, session_id: str, operation: Operation):
        """Store an operation in the database"""
        try:
            async with get_db_session() as db:
                update_record = SessionStateUpdate(
                    session_id=session_id,
                    user_id=operation.user_id,
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type.value,
                    target_path=operation.target_path,
                    old_value=operation.old_value,
                    new_value=operation.new_value,
                    operation_data=operation.metadata,
                    version=operation.version,
                    parent_version=operation.parent_version
                )
                
                db.add(update_record)
                db.commit()
        except Exception as e:
            logger.error(f"Error storing operation: {e}")
            raise
    
    async def _update_state_cache(self, session_id: str, operation: Operation):
        """Update the cached session state with an operation"""
        try:
            cache_key = f"session:{session_id}:state"
            current_state = await redis_manager.get_json(cache_key) or {}
            
            # Apply operation to state
            updated_state = await self._apply_operation_to_state(current_state, operation)
            
            # Update cache
            await redis_manager.set_json(cache_key, updated_state)
        except Exception as e:
            logger.error(f"Error updating state cache: {e}")
    
    async def _apply_operation_to_state(self, state: Dict[str, Any], operation: Operation) -> Dict[str, Any]:
        """Apply a single operation to a state dictionary"""
        try:
            if operation.operation_type == OperationType.FILTER_CHANGE:
                if "active_filters" not in state:
                    state["active_filters"] = {}
                state["active_filters"][operation.target_path] = operation.new_value
                
            elif operation.operation_type == OperationType.DASHBOARD_CONFIG:
                if "dashboard_config" not in state:
                    state["dashboard_config"] = {}
                if isinstance(operation.new_value, dict):
                    state["dashboard_config"].update(operation.new_value)
                else:
                    state["dashboard_config"][operation.target_path] = operation.new_value
                
            elif operation.operation_type == OperationType.CURSOR_MOVE:
                if "cursor_positions" not in state:
                    state["cursor_positions"] = {}
                state["cursor_positions"][operation.user_id] = operation.new_value
                
            elif operation.operation_type == OperationType.ANNOTATION_ADD:
                if "annotations" not in state:
                    state["annotations"] = []
                state["annotations"].append(operation.new_value)
                
            elif operation.operation_type == OperationType.ANNOTATION_REMOVE:
                if "annotations" in state:
                    state["annotations"] = [
                        ann for ann in state["annotations"] 
                        if ann.get("id") != operation.target_path
                    ]
            
            # Update metadata
            state["version"] = operation.version
            state["last_updated"] = operation.timestamp.isoformat()
            state["last_updated_by"] = operation.user_id
            
            return state
        except Exception as e:
            logger.error(f"Error applying operation to state: {e}")
            return state
    
    async def _apply_operations_to_state(self, initial_state: Dict[str, Any], 
                                       operations: List[Any]) -> Dict[str, Any]:
        """Apply a list of operations to rebuild state"""
        try:
            state = initial_state.copy()
            
            for db_operation in operations:
                operation = Operation(
                    operation_id=db_operation.operation_id,
                    operation_type=OperationType(db_operation.operation_type),
                    target_path=db_operation.target_path,
                    old_value=db_operation.old_value,
                    new_value=db_operation.new_value,
                    user_id=str(db_operation.user_id),
                    session_id=str(db_operation.session_id),
                    version=db_operation.version,
                    timestamp=db_operation.created_at
                )
                
                state = await self._apply_operation_to_state(state, operation)
            
            return state
        except Exception as e:
            logger.error(f"Error applying operations to state: {e}")
            return initial_state
    
    async def _rebuild_state_from_operations(self, session_id: str) -> Dict[str, Any]:
        """Rebuild session state from all operations"""
        try:
            async with get_db_session() as db:
                operations = db.query(SessionStateUpdate).filter(
                    SessionStateUpdate.session_id == session_id
                ).order_by(SessionStateUpdate.version).all()
                
                initial_state = {
                    "dashboard_config": {},
                    "active_filters": {},
                    "annotations": [],
                    "cursor_positions": {},
                    "version": 0
                }
                
                return await self._apply_operations_to_state(initial_state, operations)
        except Exception as e:
            logger.error(f"Error rebuilding state from operations: {e}")
            return {}
    
    async def _create_state_version(self, session_id: str, operation: Operation):
        """Create a new state version snapshot"""
        try:
            if session_id not in self.state_versions:
                self.state_versions[session_id] = []
            
            # Get current state
            current_state = await self.get_current_state(session_id)
            
            # Create version snapshot
            version = StateVersion(
                version=operation.version,
                operations=[operation],
                state_snapshot=current_state,
                created_by=operation.user_id
            )
            
            self.state_versions[session_id].append(version)
            
            # Keep only last 100 versions to manage memory
            if len(self.state_versions[session_id]) > 100:
                self.state_versions[session_id] = self.state_versions[session_id][-100:]
                
        except Exception as e:
            logger.error(f"Error creating state version: {e}")
    
    async def _store_conflict_resolution(self, session_id: str, conflict: ConflictInfo):
        """Store conflict resolution information"""
        try:
            # In production, this would store conflict resolution in database
            logger.info(f"Conflict resolved for session {session_id}: {conflict.conflict_id}")
        except Exception as e:
            logger.error(f"Error storing conflict resolution: {e}")

# Global operational transformation engine instance
ot_engine = OperationalTransformationEngine()