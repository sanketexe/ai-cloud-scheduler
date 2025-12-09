"""
Intelligent Cache Invalidation System

This module provides advanced cache invalidation strategies including pattern-based
invalidation, cache tags for related data, time-based expiration, and event-driven
invalidation for the FinOps platform.
"""

import asyncio
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
from .redis_config import get_redis
from .cache_service import cache_service

logger = structlog.get_logger(__name__)


class InvalidationStrategy(Enum):
    """Cache invalidation strategies"""
    IMMEDIATE = "immediate"
    LAZY = "lazy"
    TIME_BASED = "time_based"
    EVENT_DRIVEN = "event_driven"
    PATTERN_BASED = "pattern_based"


@dataclass
class CacheTag:
    """Represents a cache tag for grouping related cache entries"""
    name: str
    description: str
    ttl: Optional[int] = None
    auto_invalidate: bool = True
    related_tags: Set[str] = field(default_factory=set)


@dataclass
class InvalidationRule:
    """Defines rules for cache invalidation"""
    trigger_event: str
    target_patterns: List[str]
    strategy: InvalidationStrategy
    delay_seconds: int = 0
    cascade_tags: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None


class CacheTagManager:
    """Manages cache tags and their relationships"""
    
    def __init__(self):
        self.tags: Dict[str, CacheTag] = {}
        self.key_tags: Dict[str, Set[str]] = {}  # Maps cache keys to their tags
        self.tag_keys: Dict[str, Set[str]] = {}  # Maps tags to their cache keys
        
    def register_tag(self, tag: CacheTag):
        """Register a new cache tag"""
        self.tags[tag.name] = tag
        if tag.name not in self.tag_keys:
            self.tag_keys[tag.name] = set()
        
        logger.info("Cache tag registered", tag_name=tag.name, description=tag.description)
    
    def add_key_to_tag(self, cache_key: str, tag_name: str):
        """Associate a cache key with a tag"""
        if tag_name not in self.tags:
            logger.warning("Attempting to use unregistered tag", tag_name=tag_name)
            return
        
        if cache_key not in self.key_tags:
            self.key_tags[cache_key] = set()
        
        self.key_tags[cache_key].add(tag_name)
        self.tag_keys[tag_name].add(cache_key)
        
        logger.debug("Key added to tag", cache_key=cache_key, tag_name=tag_name)
    
    def remove_key_from_tag(self, cache_key: str, tag_name: str):
        """Remove association between cache key and tag"""
        if cache_key in self.key_tags:
            self.key_tags[cache_key].discard(tag_name)
            if not self.key_tags[cache_key]:
                del self.key_tags[cache_key]
        
        if tag_name in self.tag_keys:
            self.tag_keys[tag_name].discard(cache_key)
    
    def get_keys_by_tag(self, tag_name: str) -> Set[str]:
        """Get all cache keys associated with a tag"""
        return self.tag_keys.get(tag_name, set()).copy()
    
    def get_tags_by_key(self, cache_key: str) -> Set[str]:
        """Get all tags associated with a cache key"""
        return self.key_tags.get(cache_key, set()).copy()
    
    def get_related_tags(self, tag_name: str) -> Set[str]:
        """Get all tags related to the given tag (including cascading relationships)"""
        if tag_name not in self.tags:
            return set()
        
        related = set()
        to_process = {tag_name}
        processed = set()
        
        while to_process:
            current_tag = to_process.pop()
            if current_tag in processed:
                continue
            
            processed.add(current_tag)
            if current_tag in self.tags:
                tag_obj = self.tags[current_tag]
                related.update(tag_obj.related_tags)
                to_process.update(tag_obj.related_tags - processed)
        
        return related


class CacheInvalidationEngine:
    """Advanced cache invalidation engine with multiple strategies"""
    
    def __init__(self):
        self.tag_manager = CacheTagManager()
        self.invalidation_rules: List[InvalidationRule] = []
        self.pending_invalidations: Dict[str, datetime] = {}
        self.event_listeners: Dict[str, List[Callable]] = {}
        
        # Initialize common tags
        self._initialize_common_tags()
    
    def _initialize_common_tags(self):
        """Initialize commonly used cache tags"""
        common_tags = [
            CacheTag("cost_data", "Cost and billing data", ttl=1800),
            CacheTag("budget_data", "Budget information", ttl=900),
            CacheTag("user_data", "User profile and settings", ttl=3600),
            CacheTag("provider_data", "Cloud provider configurations", ttl=7200),
            CacheTag("optimization_data", "Optimization recommendations", ttl=3600),
            CacheTag("compliance_data", "Compliance reports and policies", ttl=1800),
            CacheTag("dashboard_data", "Dashboard widgets and metrics", ttl=600),
            CacheTag("report_data", "Generated reports", ttl=1800)
        ]
        
        for tag in common_tags:
            self.tag_manager.register_tag(tag)
        
        # Set up tag relationships
        self._setup_tag_relationships()
    
    def _setup_tag_relationships(self):
        """Set up relationships between cache tags"""
        relationships = {
            "cost_data": {"budget_data", "dashboard_data", "report_data"},
            "budget_data": {"dashboard_data", "report_data"},
            "provider_data": {"cost_data", "optimization_data"},
            "user_data": {"dashboard_data"},
            "optimization_data": {"report_data", "dashboard_data"},
            "compliance_data": {"report_data", "dashboard_data"}
        }
        
        for tag_name, related_tags in relationships.items():
            if tag_name in self.tag_manager.tags:
                self.tag_manager.tags[tag_name].related_tags.update(related_tags)
    
    def register_invalidation_rule(self, rule: InvalidationRule):
        """Register a new invalidation rule"""
        self.invalidation_rules.append(rule)
        logger.info("Invalidation rule registered", 
                   trigger_event=rule.trigger_event, 
                   strategy=rule.strategy.value)
    
    async def invalidate_by_tag(self, tag_name: str, cascade: bool = True) -> int:
        """Invalidate all cache entries associated with a tag"""
        try:
            keys_to_invalidate = self.tag_manager.get_keys_by_tag(tag_name)
            
            if cascade:
                # Get related tags and their keys
                related_tags = self.tag_manager.get_related_tags(tag_name)
                for related_tag in related_tags:
                    keys_to_invalidate.update(self.tag_manager.get_keys_by_tag(related_tag))
            
            if not keys_to_invalidate:
                logger.debug("No keys found for tag invalidation", tag_name=tag_name)
                return 0
            
            # Invalidate all keys
            redis_client = await get_redis()
            deleted_count = 0
            
            for cache_key in keys_to_invalidate:
                try:
                    result = await redis_client.delete(cache_key)
                    if result:
                        deleted_count += 1
                        # Remove key from tag associations
                        for tag in self.tag_manager.get_tags_by_key(cache_key):
                            self.tag_manager.remove_key_from_tag(cache_key, tag)
                except Exception as e:
                    logger.error("Failed to delete cache key", 
                               cache_key=cache_key, error=str(e))
            
            logger.info("Tag-based invalidation completed", 
                       tag_name=tag_name, deleted_count=deleted_count, cascade=cascade)
            return deleted_count
            
        except Exception as e:
            logger.error("Tag-based invalidation failed", 
                        tag_name=tag_name, error=str(e))
            return 0
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern"""
        try:
            redis_client = await get_redis()
            
            # Get all keys matching the pattern
            keys = await redis_client.keys(f"finops:{pattern}")
            
            if not keys:
                logger.debug("No keys found for pattern", pattern=pattern)
                return 0
            
            # Delete all matching keys
            deleted_count = await redis_client.delete(*keys)
            
            # Clean up tag associations
            for key in keys:
                for tag in self.tag_manager.get_tags_by_key(key):
                    self.tag_manager.remove_key_from_tag(key, tag)
            
            logger.info("Pattern-based invalidation completed", 
                       pattern=pattern, deleted_count=deleted_count)
            return deleted_count
            
        except Exception as e:
            logger.error("Pattern-based invalidation failed", 
                        pattern=pattern, error=str(e))
            return 0
    
    async def invalidate_by_event(self, event_name: str, event_data: Dict[str, Any] = None):
        """Trigger invalidation based on an event"""
        try:
            event_data = event_data or {}
            invalidated_count = 0
            
            # Find matching invalidation rules
            matching_rules = [rule for rule in self.invalidation_rules 
                            if rule.trigger_event == event_name]
            
            for rule in matching_rules:
                # Check condition if specified
                if rule.condition and not rule.condition(event_data):
                    continue
                
                # Apply delay if specified
                if rule.delay_seconds > 0:
                    await asyncio.sleep(rule.delay_seconds)
                
                # Execute invalidation based on strategy
                if rule.strategy == InvalidationStrategy.PATTERN_BASED:
                    for pattern in rule.target_patterns:
                        count = await self.invalidate_by_pattern(pattern)
                        invalidated_count += count
                
                elif rule.strategy == InvalidationStrategy.EVENT_DRIVEN:
                    # Invalidate cascade tags
                    for tag_name in rule.cascade_tags:
                        count = await self.invalidate_by_tag(tag_name, cascade=True)
                        invalidated_count += count
            
            # Notify event listeners
            if event_name in self.event_listeners:
                for listener in self.event_listeners[event_name]:
                    try:
                        await listener(event_name, event_data)
                    except Exception as e:
                        logger.error("Event listener failed", 
                                   event_name=event_name, error=str(e))
            
            logger.info("Event-driven invalidation completed", 
                       event_name=event_name, invalidated_count=invalidated_count)
            
        except Exception as e:
            logger.error("Event-driven invalidation failed", 
                        event_name=event_name, error=str(e))
    
    async def schedule_time_based_invalidation(self, cache_key: str, 
                                             invalidate_at: datetime):
        """Schedule time-based cache invalidation"""
        try:
            redis_client = await get_redis()
            
            # Calculate TTL in seconds
            now = datetime.utcnow()
            if invalidate_at <= now:
                # Immediate invalidation
                await redis_client.delete(cache_key)
                return
            
            ttl_seconds = int((invalidate_at - now).total_seconds())
            
            # Set expiration on the key
            await redis_client.expire(cache_key, ttl_seconds)
            
            logger.debug("Time-based invalidation scheduled", 
                        cache_key=cache_key, invalidate_at=invalidate_at)
            
        except Exception as e:
            logger.error("Failed to schedule time-based invalidation", 
                        cache_key=cache_key, error=str(e))
    
    def add_event_listener(self, event_name: str, listener: Callable):
        """Add an event listener for cache invalidation events"""
        if event_name not in self.event_listeners:
            self.event_listeners[event_name] = []
        
        self.event_listeners[event_name].append(listener)
        logger.debug("Event listener added", event_name=event_name)
    
    async def warm_cache(self, warming_functions: Dict[str, Callable]):
        """Warm cache with frequently accessed data"""
        try:
            for cache_key, warming_function in warming_functions.items():
                try:
                    # Check if key already exists
                    if await cache_service.exists(cache_key):
                        logger.debug("Cache key already exists, skipping warming", 
                                   cache_key=cache_key)
                        continue
                    
                    # Execute warming function
                    data = await warming_function()
                    
                    # Cache the result
                    await cache_service.set(cache_key, data)
                    
                    logger.debug("Cache warmed successfully", cache_key=cache_key)
                    
                except Exception as e:
                    logger.error("Cache warming failed for key", 
                               cache_key=cache_key, error=str(e))
            
            logger.info("Cache warming completed", 
                       total_keys=len(warming_functions))
            
        except Exception as e:
            logger.error("Cache warming process failed", error=str(e))


# Global cache invalidation engine
invalidation_engine = CacheInvalidationEngine()


# Common invalidation rules setup
def setup_common_invalidation_rules():
    """Set up common invalidation rules for the FinOps platform"""
    
    # Cost data invalidation rules
    invalidation_engine.register_invalidation_rule(
        InvalidationRule(
            trigger_event="cost_data_updated",
            target_patterns=["cost_data:*"],
            strategy=InvalidationStrategy.PATTERN_BASED,
            cascade_tags=["dashboard_data", "report_data"]
        )
    )
    
    # Budget invalidation rules
    invalidation_engine.register_invalidation_rule(
        InvalidationRule(
            trigger_event="budget_updated",
            target_patterns=["budget:*"],
            strategy=InvalidationStrategy.EVENT_DRIVEN,
            cascade_tags=["budget_data", "dashboard_data"]
        )
    )
    
    # Provider configuration changes
    invalidation_engine.register_invalidation_rule(
        InvalidationRule(
            trigger_event="provider_config_changed",
            target_patterns=["provider:*", "cost_data:*"],
            strategy=InvalidationStrategy.PATTERN_BASED,
            cascade_tags=["provider_data", "cost_data"]
        )
    )
    
    # User settings changes
    invalidation_engine.register_invalidation_rule(
        InvalidationRule(
            trigger_event="user_settings_changed",
            target_patterns=["user:*"],
            strategy=InvalidationStrategy.PATTERN_BASED,
            cascade_tags=["user_data", "dashboard_data"]
        )
    )


# Convenience functions for common invalidation scenarios
async def invalidate_cost_data(provider_id: str = None, date_range: str = None):
    """Invalidate cost data cache"""
    if provider_id and date_range:
        pattern = f"cost_data:{provider_id}:{date_range}:*"
    elif provider_id:
        pattern = f"cost_data:{provider_id}:*"
    else:
        pattern = "cost_data:*"
    
    return await invalidation_engine.invalidate_by_pattern(pattern)


async def invalidate_dashboard_cache(user_id: str = None):
    """Invalidate dashboard cache"""
    if user_id:
        pattern = f"dashboard:{user_id}:*"
        return await invalidation_engine.invalidate_by_pattern(pattern)
    else:
        return await invalidation_engine.invalidate_by_tag("dashboard_data")


async def invalidate_report_cache(report_type: str = None):
    """Invalidate report cache"""
    if report_type:
        pattern = f"report:{report_type}:*"
        return await invalidation_engine.invalidate_by_pattern(pattern)
    else:
        return await invalidation_engine.invalidate_by_tag("report_data")


# Initialize common rules when module is imported
setup_common_invalidation_rules()