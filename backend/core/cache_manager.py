"""
Cache Manager for Multi-Cloud Cost Engine

Provides intelligent caching strategies with multi-layer cache support,
cache statistics tracking, and automatic invalidation mechanisms.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class CacheLayer(Enum):
    """Available cache layers"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


@dataclass
class CacheStats:
    """Cache statistics tracking"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    redis_usage_mb: float = 0.0
    evictions: int = 0
    invalidations: int = 0
    last_updated: datetime = None
    
    def calculate_hit_rate(self):
        """Calculate cache hit rate"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests
        else:
            self.hit_rate = 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    layer: CacheLayer = CacheLayer.MEMORY
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def touch(self):
        """Update access information"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class MemoryCache:
    """In-memory cache implementation with LRU eviction"""
    
    def __init__(self, max_entries: int = 10000, max_size_mb: int = 100):
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_size_bytes = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key not in self.entries:
            return None
        
        entry = self.entries[key]
        
        # Check expiration
        if entry.is_expired():
            await self.delete(key)
            return None
        
        # Update access order for LRU
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        entry.touch()
        return entry.value
    
    async def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Set value in memory cache"""
        try:
            # Serialize to calculate size
            serialized = json.dumps(value, default=str)
            size_bytes = len(serialized.encode('utf-8'))
            
            # Check if we need to evict entries
            await self._ensure_capacity(size_bytes)
            
            # Create cache entry
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                layer=CacheLayer.MEMORY
            )
            
            # Remove old entry if exists
            if key in self.entries:
                await self.delete(key)
            
            # Add new entry
            self.entries[key] = entry
            self.access_order.append(key)
            self.current_size_bytes += size_bytes
            
            return True
            
        except Exception as e:
            logger.error("Memory cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from memory cache"""
        if key not in self.entries:
            return False
        
        entry = self.entries[key]
        self.current_size_bytes -= entry.size_bytes
        
        del self.entries[key]
        if key in self.access_order:
            self.access_order.remove(key)
        
        return True
    
    async def clear(self):
        """Clear all entries from memory cache"""
        self.entries.clear()
        self.access_order.clear()
        self.current_size_bytes = 0
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry"""
        # Check entry count limit
        while len(self.entries) >= self.max_entries:
            await self._evict_lru()
        
        # Check size limit
        while (self.current_size_bytes + new_entry_size) > self.max_size_bytes:
            await self._evict_lru()
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_order:
            return
        
        lru_key = self.access_order[0]
        await self.delete(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        return {
            'entry_count': len(self.entries),
            'size_bytes': self.current_size_bytes,
            'size_mb': self.current_size_bytes / (1024 * 1024),
            'max_entries': self.max_entries,
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }


class RedisCache:
    """Redis cache implementation"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", key_prefix: str = "mcce:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_client = None
        self.connected = False
    
    async def connect(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping Redis cache")
            return False
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.connected = True
            logger.info("Redis cache connected", url=self.redis_url)
            return True
        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            self.connected = False
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.connected:
            return None
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            data = await self.redis_client.get(redis_key)
            
            if data is None:
                return None
            
            # Deserialize JSON data
            return json.loads(data)
            
        except Exception as e:
            logger.error("Redis get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Set value in Redis cache"""
        if not self.connected:
            return False
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            serialized = json.dumps(value, default=str)
            
            await self.redis_client.setex(redis_key, ttl_seconds, serialized)
            return True
            
        except Exception as e:
            logger.error("Redis set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from Redis cache"""
        if not self.connected:
            return False
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            result = await self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error("Redis delete failed", key=key, error=str(e))
            return False
    
    async def clear(self):
        """Clear all entries with our prefix"""
        if not self.connected:
            return
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error("Redis clear failed", error=str(e))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.connected:
            return {}
        
        try:
            info = await self.redis_client.info('memory')
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected': self.connected
            }
        except Exception as e:
            logger.error("Redis stats failed", error=str(e))
            return {'connected': False}


class CacheManager:
    """
    Multi-layer cache manager with intelligent caching strategies.
    
    Provides memory and Redis caching with automatic failover,
    cache statistics tracking, and intelligent invalidation.
    """
    
    def __init__(self, redis_url: str = None, enable_redis: bool = True):
        self.memory_cache = MemoryCache()
        self.redis_cache = RedisCache(redis_url) if enable_redis and redis_url else None
        self.stats = CacheStats()
        self.invalidation_patterns: Dict[str, List[str]] = {}
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.memory_first = True  # Check memory cache first
        self.write_through = True  # Write to all layers
        
    async def initialize(self):
        """Initialize cache manager and connections"""
        logger.info("Initializing cache manager")
        
        if self.redis_cache:
            await self.redis_cache.connect()
        
        self.stats.last_updated = datetime.utcnow()
        logger.info("Cache manager initialized")
    
    async def get_or_compute_comparison(self, workload_hash: str, compute_func: Callable) -> Any:
        """
        Get cached comparison result or compute and cache it.
        
        Args:
            workload_hash: Unique hash for the workload specification
            compute_func: Function to compute the result if not cached
            
        Returns:
            Comparison result (cached or computed)
        """
        start_time = time.time()
        
        try:
            # Try to get from cache
            cache_key = f"comparison:{workload_hash}"
            cached_result = await self.get(cache_key)
            
            if cached_result is not None:
                response_time = (time.time() - start_time) * 1000
                self._update_stats(hit=True, response_time_ms=response_time)
                
                logger.info(
                    "Cache hit for comparison",
                    workload_hash=workload_hash,
                    response_time_ms=response_time
                )
                
                return cached_result
            
            # Cache miss - compute result
            logger.info("Cache miss for comparison", workload_hash=workload_hash)
            
            result = await compute_func()
            
            # Cache the computed result
            await self.set(cache_key, result, self.default_ttl)
            
            response_time = (time.time() - start_time) * 1000
            self._update_stats(hit=False, response_time_ms=response_time)
            
            logger.info(
                "Computed and cached comparison",
                workload_hash=workload_hash,
                response_time_ms=response_time
            )
            
            return result
            
        except Exception as e:
            logger.error("Get or compute comparison failed", error=str(e))
            # Fallback to direct computation
            return await compute_func()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks all layers)"""
        self.stats.total_requests += 1
        
        # Check memory cache first
        if self.memory_first:
            result = await self.memory_cache.get(key)
            if result is not None:
                return result
        
        # Check Redis cache
        if self.redis_cache and self.redis_cache.connected:
            result = await self.redis_cache.get(key)
            if result is not None:
                # Populate memory cache for faster future access
                await self.memory_cache.set(key, result, self.default_ttl)
                return result
        
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        """Set value in cache (writes to all layers if write_through enabled)"""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        success = True
        
        # Write to memory cache
        memory_success = await self.memory_cache.set(key, value, ttl_seconds)
        if not memory_success:
            success = False
        
        # Write to Redis cache if write_through enabled
        if self.write_through and self.redis_cache and self.redis_cache.connected:
            redis_success = await self.redis_cache.set(key, value, ttl_seconds)
            if not redis_success:
                success = False
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache layers"""
        memory_success = await self.memory_cache.delete(key)
        
        redis_success = True
        if self.redis_cache and self.redis_cache.connected:
            redis_success = await self.redis_cache.delete(key)
        
        return memory_success or redis_success
    
    async def invalidate_pricing_cache(self, provider: str, service: str = None):
        """
        Invalidate pricing cache for specific provider/service.
        
        Args:
            provider: Cloud provider (aws, gcp, azure)
            service: Specific service (optional, invalidates all if None)
        """
        logger.info("Invalidating pricing cache", provider=provider, service=service)
        
        try:
            # Build invalidation pattern
            if service:
                pattern = f"pricing:{provider}:{service}:*"
            else:
                pattern = f"pricing:{provider}:*"
            
            # Track invalidation
            self.stats.invalidations += 1
            
            # For memory cache, we need to iterate through keys
            keys_to_delete = []
            for key in self.memory_cache.entries.keys():
                if self._matches_pattern(key, pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                await self.memory_cache.delete(key)
            
            # For Redis, we can use pattern matching
            if self.redis_cache and self.redis_cache.connected:
                try:
                    redis_pattern = f"{self.redis_cache.key_prefix}{pattern}"
                    keys = await self.redis_cache.redis_client.keys(redis_pattern)
                    if keys:
                        await self.redis_cache.redis_client.delete(*keys)
                except Exception as e:
                    logger.error("Redis invalidation failed", error=str(e))
            
            logger.info(
                "Pricing cache invalidated",
                provider=provider,
                service=service,
                keys_deleted=len(keys_to_delete)
            )
            
        except Exception as e:
            logger.error("Cache invalidation failed", error=str(e))
    
    def get_cache_statistics(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        # Update hit rate
        self.stats.calculate_hit_rate()
        
        # Get memory stats
        memory_stats = self.memory_cache.get_stats()
        self.stats.memory_usage_mb = memory_stats['size_mb']
        
        # Get Redis stats if available
        if self.redis_cache and self.redis_cache.connected:
            asyncio.create_task(self._update_redis_stats())
        
        self.stats.last_updated = datetime.utcnow()
        return self.stats
    
    async def _update_redis_stats(self):
        """Update Redis statistics"""
        try:
            redis_stats = await self.redis_cache.get_stats()
            self.stats.redis_usage_mb = redis_stats.get('used_memory_mb', 0)
        except Exception as e:
            logger.error("Failed to update Redis stats", error=str(e))
    
    def _update_stats(self, hit: bool, response_time_ms: float):
        """Update cache statistics"""
        if hit:
            self.stats.cache_hits += 1
        else:
            self.stats.cache_misses += 1
        
        # Update average response time
        total_responses = self.stats.cache_hits + self.stats.cache_misses
        if total_responses > 0:
            current_avg = self.stats.avg_response_time_ms
            self.stats.avg_response_time_ms = (
                (current_avg * (total_responses - 1) + response_time_ms) / total_responses
            )
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches invalidation pattern"""
        # Simple pattern matching (supports * wildcard)
        if '*' not in pattern:
            return key == pattern
        
        # Convert pattern to regex-like matching
        pattern_parts = pattern.split('*')
        
        # Check if key starts with first part
        if not key.startswith(pattern_parts[0]):
            return False
        
        # Check if key ends with last part (if not empty)
        if len(pattern_parts) > 1 and pattern_parts[-1]:
            if not key.endswith(pattern_parts[-1]):
                return False
        
        return True
    
    async def warm_cache(self, workload_specs: List[Dict[str, Any]]):
        """
        Warm cache with common workload specifications.
        
        Args:
            workload_specs: List of workload specifications to pre-compute
        """
        logger.info("Starting cache warming", workload_count=len(workload_specs))
        
        warming_tasks = []
        for spec in workload_specs:
            # Generate workload hash
            workload_hash = hashlib.md5(str(spec).encode()).hexdigest()
            
            # Create warming task
            async def warm_workload(spec_data, spec_hash):
                try:
                    # Mock computation for warming (in production, call actual comparison)
                    await asyncio.sleep(0.1)  # Simulate computation
                    result = {'workload': spec_data, 'cached_at': datetime.utcnow().isoformat()}
                    
                    cache_key = f"comparison:{spec_hash}"
                    await self.set(cache_key, result, self.default_ttl)
                    
                except Exception as e:
                    logger.error("Cache warming failed for workload", error=str(e))
            
            warming_tasks.append(warm_workload(spec, workload_hash))
        
        # Execute warming tasks in parallel
        await asyncio.gather(*warming_tasks, return_exceptions=True)
        
        logger.info("Cache warming completed", workload_count=len(workload_specs))
    
    async def cleanup_expired_entries(self):
        """Clean up expired entries from all cache layers"""
        logger.info("Starting cache cleanup")
        
        # Memory cache cleanup
        expired_keys = []
        for key, entry in self.memory_cache.entries.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.memory_cache.delete(key)
        
        logger.info("Cache cleanup completed", expired_entries=len(expired_keys))
    
    async def shutdown(self):
        """Shutdown cache manager and close connections"""
        logger.info("Shutting down cache manager")
        
        if self.redis_cache and self.redis_cache.redis_client:
            await self.redis_cache.redis_client.close()
        
        await self.memory_cache.clear()
        
        logger.info("Cache manager shutdown completed")