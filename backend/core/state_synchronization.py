"""
Real-Time State Synchronization Engine for Collaborative FinOps Workspace

This module provides sub-200ms state synchronization across participants,
cursor tracking, presence indicators, and dashboard view synchronization.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

from .operational_transformation import (
    OperationalTransformationEngine, Operation, OperationType, ot_engine
)
from .collaboration_models import SessionParticipant, ParticipantStatus
from .collaborative_session_manager import session_manager
from .participant_manager import participant_manager
from .redis_config import redis_manager
from .database import get_db_session

logger = logging.getLogger(__name__)

class SyncEventType(Enum):
    """Types of synchronization events"""
    STATE_UPDATE = "state_update"
    CURSOR_UPDATE = "cursor_update"
    PRESENCE_UPDATE = "presence_update"
    VIEW_CHANGE = "view_change"
    FILTER_CHANGE = "filter_change"
    DASHBOARD_CONFIG = "dashboard_config"
    ANNOTATION_UPDATE = "annotation_update"
    PARTICIPANT_JOIN = "participant_join"
    PARTICIPANT_LEAVE = "participant_leave"

@dataclass
class SyncEvent:
    """Synchronization event data"""
    event_id: str
    event_type: SyncEventType
    session_id: str
    user_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    target_participants: Optional[List[str]] = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical

@dataclass
class CursorPosition:
    """Cursor position and tracking data"""
    user_id: str
    session_id: str
    x: float
    y: float
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    viewport_width: int = 1920
    viewport_height: int = 1080
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

@dataclass
class PresenceIndicator:
    """Presence indicator data"""
    user_id: str
    display_name: str
    status: ParticipantStatus
    cursor_position: Optional[CursorPosition] = None
    current_view: str = "dashboard"
    is_typing: bool = False
    last_active: datetime = field(default_factory=datetime.utcnow)
    connection_quality: str = "good"  # good, fair, poor
    avatar_color: str = "#007bff"

@dataclass
class ViewState:
    """Dashboard view state"""
    view_id: str
    view_type: str  # dashboard, reports, settings, etc.
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_config: Dict[str, Any] = field(default_factory=dict)
    display_config: Dict[str, Any] = field(default_factory=dict)
    scroll_position: Dict[str, float] = field(default_factory=dict)
    selected_elements: List[str] = field(default_factory=list)
    zoom_level: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SynchronizationMetrics:
    """Metrics for synchronization performance"""
    session_id: str
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    sync_events_count: int = 0
    failed_syncs: int = 0
    participants_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

class StateSynchronizationEngine:
    """
    Real-time state synchronization engine providing sub-200ms synchronization
    across all session participants with cursor tracking and presence indicators
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, Set[str]] = {}  # session_id -> participant_ids
        self.cursor_positions: Dict[str, CursorPosition] = {}  # user_id -> cursor
        self.presence_indicators: Dict[str, PresenceIndicator] = {}  # user_id -> presence
        self.view_states: Dict[str, ViewState] = {}  # session_id -> view_state
        self.sync_metrics: Dict[str, SynchronizationMetrics] = {}  # session_id -> metrics
        
        # Performance optimization
        self.sync_queues: Dict[str, asyncio.Queue] = {}  # session_id -> event queue
        self.sync_workers: Dict[str, asyncio.Task] = {}  # session_id -> worker task
        self.batch_timers: Dict[str, asyncio.Task] = {}  # session_id -> batch timer
        
        # Event handlers
        self.event_handlers: Dict[SyncEventType, List[Callable]] = {}
        
        # Performance targets
        self.target_latency_ms = 200
        self.batch_size = 10
        self.batch_timeout_ms = 50
        
    async def initialize_session_sync(self, session_id: str) -> bool:
        """
        Initialize synchronization for a collaborative session
        
        Args:
            session_id: ID of the session to initialize
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already initialized for sync")
                return True
            
            # Initialize session data structures
            self.active_sessions[session_id] = set()
            self.sync_queues[session_id] = asyncio.Queue()
            self.sync_metrics[session_id] = SynchronizationMetrics(session_id=session_id)
            
            # Initialize view state
            self.view_states[session_id] = ViewState(
                view_id=f"{session_id}_main",
                view_type="dashboard"
            )
            
            # Start sync worker for this session
            self.sync_workers[session_id] = asyncio.create_task(
                self._sync_worker(session_id)
            )
            
            # Initialize Redis channels for real-time communication
            await self._setup_redis_channels(session_id)
            
            logger.info(f"Initialized synchronization for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing session sync: {e}")
            return False
    
    async def add_participant_to_sync(self, session_id: str, user_id: str, 
                                    display_name: str) -> bool:
        """
        Add a participant to session synchronization
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            display_name: Display name of the user
            
        Returns:
            bool: True if participant was added successfully
        """
        try:
            if session_id not in self.active_sessions:
                await self.initialize_session_sync(session_id)
            
            # Add to active participants
            self.active_sessions[session_id].add(user_id)
            
            # Initialize presence indicator
            self.presence_indicators[user_id] = PresenceIndicator(
                user_id=user_id,
                display_name=display_name,
                status=ParticipantStatus.ACTIVE,
                avatar_color=self._generate_avatar_color(user_id)
            )
            
            # Initialize cursor position
            self.cursor_positions[user_id] = CursorPosition(
                user_id=user_id,
                session_id=session_id,
                x=0, y=0
            )
            
            # Update metrics
            self.sync_metrics[session_id].participants_count = len(self.active_sessions[session_id])
            
            # Broadcast participant join event
            join_event = SyncEvent(
                event_id=f"join_{user_id}_{int(time.time() * 1000)}",
                event_type=SyncEventType.PARTICIPANT_JOIN,
                session_id=session_id,
                user_id=user_id,
                data={
                    "display_name": display_name,
                    "presence": self.presence_indicators[user_id].__dict__,
                    "cursor": self.cursor_positions[user_id].__dict__
                },
                priority=3
            )
            
            await self._queue_sync_event(session_id, join_event)
            
            logger.info(f"Added participant {user_id} to session {session_id} sync")
            return True
            
        except Exception as e:
            logger.error(f"Error adding participant to sync: {e}")
            return False
    
    async def remove_participant_from_sync(self, session_id: str, user_id: str) -> bool:
        """
        Remove a participant from session synchronization
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            
        Returns:
            bool: True if participant was removed successfully
        """
        try:
            if session_id not in self.active_sessions:
                return True
            
            # Remove from active participants
            self.active_sessions[session_id].discard(user_id)
            
            # Clean up presence and cursor data
            if user_id in self.presence_indicators:
                del self.presence_indicators[user_id]
            if user_id in self.cursor_positions:
                del self.cursor_positions[user_id]
            
            # Update metrics
            self.sync_metrics[session_id].participants_count = len(self.active_sessions[session_id])
            
            # Broadcast participant leave event
            leave_event = SyncEvent(
                event_id=f"leave_{user_id}_{int(time.time() * 1000)}",
                event_type=SyncEventType.PARTICIPANT_LEAVE,
                session_id=session_id,
                user_id=user_id,
                data={"user_id": user_id},
                priority=3
            )
            
            await self._queue_sync_event(session_id, leave_event)
            
            # Clean up session if no participants
            if not self.active_sessions[session_id]:
                await self._cleanup_session_sync(session_id)
            
            logger.info(f"Removed participant {user_id} from session {session_id} sync")
            return True
            
        except Exception as e:
            logger.error(f"Error removing participant from sync: {e}")
            return False
    
    async def sync_state_update(self, session_id: str, user_id: str, 
                              operation: Operation) -> bool:
        """
        Synchronize a state update across all participants with sub-200ms latency
        
        Args:
            session_id: ID of the session
            user_id: ID of the user making the update
            operation: Operation to synchronize
            
        Returns:
            bool: True if synchronization was successful
        """
        try:
            start_time = time.time()
            
            # Apply operation through OT engine
            transform_result = await ot_engine.apply_operation(session_id, operation)
            
            # Create sync event
            sync_event = SyncEvent(
                event_id=f"state_{operation.operation_id}",
                event_type=SyncEventType.STATE_UPDATE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "operation": transform_result.transformed_operation.__dict__,
                    "conflicts": transform_result.conflicts_detected,
                    "resolution_strategy": transform_result.resolution_strategy.value
                },
                priority=2
            )
            
            # Queue for immediate synchronization
            await self._queue_sync_event(session_id, sync_event)
            
            # Update performance metrics
            latency_ms = (time.time() - start_time) * 1000
            await self._update_sync_metrics(session_id, latency_ms)
            
            # Check if we're meeting performance targets
            if latency_ms > self.target_latency_ms:
                logger.warning(f"Sync latency {latency_ms:.2f}ms exceeds target {self.target_latency_ms}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Error syncing state update: {e}")
            await self._update_sync_metrics(session_id, 0, failed=True)
            return False
    
    async def update_cursor_position(self, session_id: str, user_id: str, 
                                   x: float, y: float, element_id: Optional[str] = None,
                                   element_type: Optional[str] = None) -> bool:
        """
        Update and synchronize cursor position with real-time tracking
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            x: X coordinate
            y: Y coordinate
            element_id: Optional ID of the element under cursor
            element_type: Optional type of the element under cursor
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Update cursor position
            if user_id in self.cursor_positions:
                cursor = self.cursor_positions[user_id]
                cursor.x = x
                cursor.y = y
                cursor.element_id = element_id
                cursor.element_type = element_type
                cursor.timestamp = datetime.utcnow()
                cursor.is_active = True
            else:
                cursor = CursorPosition(
                    user_id=user_id,
                    session_id=session_id,
                    x=x, y=y,
                    element_id=element_id,
                    element_type=element_type
                )
                self.cursor_positions[user_id] = cursor
            
            # Create cursor update event
            cursor_event = SyncEvent(
                event_id=f"cursor_{user_id}_{int(time.time() * 1000)}",
                event_type=SyncEventType.CURSOR_UPDATE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "cursor": cursor.__dict__,
                    "element_info": {
                        "id": element_id,
                        "type": element_type
                    }
                },
                priority=1  # Low priority for cursor updates
            )
            
            # Queue for synchronization
            await self._queue_sync_event(session_id, cursor_event)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating cursor position: {e}")
            return False
    
    async def update_presence_indicator(self, session_id: str, user_id: str,
                                      status: ParticipantStatus,
                                      current_view: Optional[str] = None,
                                      is_typing: Optional[bool] = None) -> bool:
        """
        Update and synchronize presence indicators
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            status: Participant status
            current_view: Current view/page
            is_typing: Whether user is typing
            
        Returns:
            bool: True if update was successful
        """
        try:
            if user_id not in self.presence_indicators:
                logger.warning(f"Presence indicator not found for user {user_id}")
                return False
            
            # Update presence data
            presence = self.presence_indicators[user_id]
            presence.status = status
            presence.last_active = datetime.utcnow()
            
            if current_view is not None:
                presence.current_view = current_view
            if is_typing is not None:
                presence.is_typing = is_typing
            
            # Update connection quality based on activity
            presence.connection_quality = self._calculate_connection_quality(user_id)
            
            # Create presence update event
            presence_event = SyncEvent(
                event_id=f"presence_{user_id}_{int(time.time() * 1000)}",
                event_type=SyncEventType.PRESENCE_UPDATE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "presence": presence.__dict__
                },
                priority=2
            )
            
            # Queue for synchronization
            await self._queue_sync_event(session_id, presence_event)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating presence indicator: {e}")
            return False
    
    async def sync_view_change(self, session_id: str, user_id: str, 
                             view_type: str, view_config: Dict[str, Any]) -> bool:
        """
        Synchronize dashboard view changes across participants
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            view_type: Type of view (dashboard, reports, etc.)
            view_config: View configuration data
            
        Returns:
            bool: True if synchronization was successful
        """
        try:
            # Update view state
            if session_id in self.view_states:
                view_state = self.view_states[session_id]
                view_state.view_type = view_type
                view_state.display_config.update(view_config)
                view_state.timestamp = datetime.utcnow()
            
            # Create view change operation
            operation = Operation(
                operation_type=OperationType.VIEW_CHANGE,
                target_path=f"view_{view_type}",
                new_value=view_config,
                user_id=user_id,
                session_id=session_id
            )
            
            # Synchronize through state update
            return await self.sync_state_update(session_id, user_id, operation)
            
        except Exception as e:
            logger.error(f"Error syncing view change: {e}")
            return False
    
    async def sync_filter_change(self, session_id: str, user_id: str,
                               filter_path: str, filter_value: Any) -> bool:
        """
        Synchronize filter changes with real-time updates
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            filter_path: Path to the filter
            filter_value: New filter value
            
        Returns:
            bool: True if synchronization was successful
        """
        try:
            # Update local view state
            if session_id in self.view_states:
                self.view_states[session_id].filters[filter_path] = filter_value
            
            # Create filter change operation
            operation = Operation(
                operation_type=OperationType.FILTER_CHANGE,
                target_path=filter_path,
                new_value=filter_value,
                user_id=user_id,
                session_id=session_id
            )
            
            # Synchronize through state update
            return await self.sync_state_update(session_id, user_id, operation)
            
        except Exception as e:
            logger.error(f"Error syncing filter change: {e}")
            return False
    
    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete synchronized session state
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dict containing complete session state
        """
        try:
            # Get base state from OT engine
            base_state = await ot_engine.get_current_state(session_id)
            
            # Add real-time synchronization data
            sync_state = {
                **base_state,
                "participants": {
                    user_id: self.presence_indicators[user_id].__dict__
                    for user_id in self.active_sessions.get(session_id, set())
                    if user_id in self.presence_indicators
                },
                "cursors": {
                    user_id: self.cursor_positions[user_id].__dict__
                    for user_id in self.active_sessions.get(session_id, set())
                    if user_id in self.cursor_positions
                },
                "view_state": self.view_states.get(session_id, ViewState(
                    view_id=f"{session_id}_main",
                    view_type="dashboard"
                )).__dict__,
                "sync_metrics": self.sync_metrics.get(session_id, SynchronizationMetrics(
                    session_id=session_id
                )).__dict__
            }
            
            return sync_state
            
        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            return {}
    
    async def get_sync_performance_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get synchronization performance metrics
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dict containing performance metrics
        """
        try:
            if session_id not in self.sync_metrics:
                return {}
            
            metrics = self.sync_metrics[session_id]
            return {
                "session_id": session_id,
                "average_latency_ms": metrics.average_latency_ms,
                "max_latency_ms": metrics.max_latency_ms,
                "min_latency_ms": metrics.min_latency_ms if metrics.min_latency_ms != float('inf') else 0,
                "sync_events_count": metrics.sync_events_count,
                "failed_syncs": metrics.failed_syncs,
                "participants_count": metrics.participants_count,
                "success_rate": (metrics.sync_events_count - metrics.failed_syncs) / max(metrics.sync_events_count, 1),
                "target_latency_ms": self.target_latency_ms,
                "meeting_target": metrics.average_latency_ms <= self.target_latency_ms,
                "last_updated": metrics.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sync metrics: {e}")
            return {}
    
    async def cleanup_session_sync(self, session_id: str):
        """Clean up synchronization resources for a session"""
        await self._cleanup_session_sync(session_id)
    
    # Private methods
    
    async def _queue_sync_event(self, session_id: str, event: SyncEvent):
        """Queue a synchronization event for processing"""
        try:
            if session_id in self.sync_queues:
                await self.sync_queues[session_id].put(event)
            else:
                logger.warning(f"No sync queue for session {session_id}")
        except Exception as e:
            logger.error(f"Error queuing sync event: {e}")
    
    async def _sync_worker(self, session_id: str):
        """Worker task for processing synchronization events"""
        try:
            queue = self.sync_queues[session_id]
            batch = []
            
            while session_id in self.active_sessions:
                try:
                    # Wait for events with timeout for batching
                    event = await asyncio.wait_for(
                        queue.get(), 
                        timeout=self.batch_timeout_ms / 1000
                    )
                    batch.append(event)
                    
                    # Process batch when full or on timeout
                    if len(batch) >= self.batch_size:
                        await self._process_sync_batch(session_id, batch)
                        batch = []
                        
                except asyncio.TimeoutError:
                    # Process partial batch on timeout
                    if batch:
                        await self._process_sync_batch(session_id, batch)
                        batch = []
                        
                except Exception as e:
                    logger.error(f"Error in sync worker: {e}")
                    
        except Exception as e:
            logger.error(f"Sync worker error for session {session_id}: {e}")
    
    async def _process_sync_batch(self, session_id: str, events: List[SyncEvent]):
        """Process a batch of synchronization events"""
        try:
            if not events:
                return
            
            # Sort events by priority (higher priority first)
            events.sort(key=lambda e: e.priority, reverse=True)
            
            # Broadcast events to participants
            for event in events:
                await self._broadcast_sync_event(session_id, event)
                
        except Exception as e:
            logger.error(f"Error processing sync batch: {e}")
    
    async def _broadcast_sync_event(self, session_id: str, event: SyncEvent):
        """Broadcast synchronization event to session participants"""
        try:
            if session_id not in self.active_sessions:
                return
            
            participants = self.active_sessions[session_id]
            if event.target_participants:
                participants = participants.intersection(set(event.target_participants))
            
            # Prepare event data for broadcast
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "session_id": session_id,
                "user_id": event.user_id,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "priority": event.priority
            }
            
            # Broadcast via Redis pub/sub for real-time delivery
            for participant_id in participants:
                try:
                    channel = f"session:{session_id}:sync:{participant_id}"
                    await redis_manager.publish(channel, event_data)
                except Exception as e:
                    logger.error(f"Failed to broadcast to participant {participant_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error broadcasting sync event: {e}")
    
    async def _setup_redis_channels(self, session_id: str):
        """Set up Redis channels for real-time communication"""
        try:
            # Set up session-wide channels
            await redis_manager.setup_pubsub_channel(f"session:{session_id}:sync")
            await redis_manager.setup_pubsub_channel(f"session:{session_id}:presence")
            await redis_manager.setup_pubsub_channel(f"session:{session_id}:cursors")
            
        except Exception as e:
            logger.error(f"Error setting up Redis channels: {e}")
    
    async def _update_sync_metrics(self, session_id: str, latency_ms: float, failed: bool = False):
        """Update synchronization performance metrics"""
        try:
            if session_id not in self.sync_metrics:
                return
            
            metrics = self.sync_metrics[session_id]
            
            if failed:
                metrics.failed_syncs += 1
            else:
                metrics.sync_events_count += 1
                
                # Update latency metrics
                if latency_ms > 0:
                    if metrics.sync_events_count == 1:
                        metrics.average_latency_ms = latency_ms
                        metrics.min_latency_ms = latency_ms
                        metrics.max_latency_ms = latency_ms
                    else:
                        # Running average
                        metrics.average_latency_ms = (
                            (metrics.average_latency_ms * (metrics.sync_events_count - 1) + latency_ms) /
                            metrics.sync_events_count
                        )
                        metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
                        metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
            
            metrics.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating sync metrics: {e}")
    
    def _generate_avatar_color(self, user_id: str) -> str:
        """Generate a consistent avatar color for a user"""
        colors = [
            "#007bff", "#28a745", "#dc3545", "#ffc107", "#17a2b8",
            "#6f42c1", "#e83e8c", "#fd7e14", "#20c997", "#6c757d"
        ]
        return colors[hash(user_id) % len(colors)]
    
    def _calculate_connection_quality(self, user_id: str) -> str:
        """Calculate connection quality based on user activity"""
        try:
            if user_id not in self.presence_indicators:
                return "unknown"
            
            presence = self.presence_indicators[user_id]
            time_since_active = datetime.utcnow() - presence.last_active
            
            if time_since_active.total_seconds() < 10:
                return "good"
            elif time_since_active.total_seconds() < 30:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error calculating connection quality: {e}")
            return "unknown"
    
    async def _cleanup_session_sync(self, session_id: str):
        """Clean up synchronization resources for a session"""
        try:
            # Stop sync worker
            if session_id in self.sync_workers:
                self.sync_workers[session_id].cancel()
                del self.sync_workers[session_id]
            
            # Clean up data structures
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.sync_queues:
                del self.sync_queues[session_id]
            if session_id in self.view_states:
                del self.view_states[session_id]
            if session_id in self.sync_metrics:
                del self.sync_metrics[session_id]
            
            # Clean up batch timers
            if session_id in self.batch_timers:
                self.batch_timers[session_id].cancel()
                del self.batch_timers[session_id]
            
            logger.info(f"Cleaned up synchronization for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session sync: {e}")

# Global state synchronization engine instance
sync_engine = StateSynchronizationEngine()