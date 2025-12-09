"""
Redis Cache Service with Automatic Serialization

This module provides a comprehensive caching service with automatic JSON serialization,
cache decorators, key generation, and namespacing for the FinOps platform.
"""

import json
import pickle
import hashlib
import functools
from typing import Any, Optional, Dict, List, Union, Callable, Type, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pydantic import BaseModel
import structlog
from .redis_config import get_redis, redis_manager

logger = structlog.get_logger(__name__)

T = TypeVar('T')


@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheKeyGenerator:
    """Generates consistent cache keys with namespacing"""
    
    def __init__(self, namespace: str = "finops"):
        self.namespace = namespace
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        # Create a consistent string representation
        key_parts = []
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (dict, list, tuple)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            for key, value in sorted_kwargs:
                if isinstance(value, (str, int, float, bool)):
                    key_parts.append(f"{key}:{value}")
                elif isinstance(value, (dict, list, tuple)):
                    key_parts.append(f"{key}:{json.dumps(value, sort_keys=True)}")
                else:
                    key_parts.append(f"{key}:{str(value)}")
        
        # Create hash of the key parts for consistent length
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{self.namespace}:{key_hash}"
    
    def generate_pattern_key(self, pattern: str) -> str:
        """Generate a pattern key for bulk operations"""
        return f"{self.namespace}:{pattern}"


class CacheSerializer:
    """Handles serialization and deserialization of cached objects"""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Serialize object to bytes"""
        try:
            if isinstance(obj, (str, int, float, bool)):
                return json.dumps(obj).encode('utf-8')
            elif isinstance(obj, BaseModel):
                return json.dumps(obj.model_dump()).encode('utf-8')
            elif isinstance(obj, dict):
                return json.dumps(obj).encode('utf-8')
            elif hasattr(obj, '__dict__'):
                # Try to serialize as dict first
                try:
                    if hasattr(obj, 'model_dump'):  # Pydantic model
                        return json.dumps(obj.model_dump()).encode('utf-8')
                    elif hasattr(obj, '__dataclass_fields__'):  # Dataclass
                        return json.dumps(asdict(obj)).encode('utf-8')
                    else:
                        return json.dumps(obj.__dict__).encode('utf-8')
                except (TypeError, ValueError):
                    # Fall back to pickle for complex objects
                    return pickle.dumps(obj)
            else:
                # Use pickle for complex objects
                return pickle.dumps(obj)
        except Exception as e:
            logger.error("Serialization failed", error=str(e), obj_type=type(obj).__name__)
            raise
    
    @staticmethod
    def deserialize(data: bytes, target_type: Optional[Type[T]] = None) -> Any:
        """Deserialize bytes to object"""
        try:
            # Try JSON first
            try:
                json_data = json.loads(data.decode('utf-8'))
                
                if target_type:
                    if issubclass(target_type, BaseModel):
                        return target_type.model_validate(json_data)
                    elif hasattr(target_type, '__dataclass_fields__'):
                        return target_type(**json_data)
                
                return json_data
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error("Deserialization failed", error=str(e))
            raise


class CacheService:
    """Redis-based caching service with automatic serialization"""
    
    def __init__(self, namespace: str = "finops", default_ttl: int = 3600):
        self.namespace = namespace
        self.default_ttl = default_ttl
        self.key_generator = CacheKeyGenerator(namespace)
        self.serializer = CacheSerializer()
        self.stats = CacheStats()
        
    async def get(self, key: str, target_type: Optional[Type[T]] = None) -> Optional[T]:
        """Get value from cache with automatic deserialization"""
        try:
            redis_client = await get_redis()
            cache_key = self.key_generator.generate_key(key)
            
            data = await redis_client.get(cache_key)
            
            if data is None:
                self.stats.misses += 1
                logger.debug("Cache miss", key=cache_key)
                return None
            
            self.stats.hits += 1
            result = self.serializer.deserialize(data, target_type)
            logger.debug("Cache hit", key=cache_key, type=type(result).__name__)
            return result
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with automatic serialization"""
        try:
            redis_client = await get_redis()
            cache_key = self.key_generator.generate_key(key)
            
            serialized_data = self.serializer.serialize(value)
            ttl_seconds = ttl or self.default_ttl
            
            result = await redis_client.setex(cache_key, ttl_seconds, serialized_data)
            
            if result:
                self.stats.sets += 1
                logger.debug("Cache set successful", key=cache_key, ttl=ttl_seconds)
            
            return bool(result)
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            redis_client = await get_redis()
            cache_key = self.key_generator.generate_key(key)
            
            result = await redis_client.delete(cache_key)
            
            if result:
                self.stats.deletes += 1
                logger.debug("Cache delete successful", key=cache_key)
            
            return bool(result)
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await get_redis()
            cache_key = self.key_generator.generate_key(key)
            
            result = await redis_client.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error("Cache exists check failed", key=key, error=str(e))
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for a key"""
        try:
            redis_client = await get_redis()
            cache_key = self.key_generator.generate_key(key)
            
            ttl = await redis_client.ttl(cache_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            logger.error("Cache TTL check failed", key=key, error=str(e))
            return None
    
    async def extend_ttl(self, key: str, ttl: int) -> bool:
        """Extend TTL for an existing key"""
        try:
            redis_client = await get_redis()
            cache_key = self.key_generator.generate_key(key)
            
            result = await redis_client.expire(cache_key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error("Cache TTL extension failed", key=key, error=str(e))
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern"""
        try:
            redis_client = await get_redis()
            pattern_key = self.key_generator.generate_pattern_key(pattern)
            
            keys = await redis_client.keys(pattern_key)
            if keys:
                deleted_count = await redis_client.delete(*keys)
                self.stats.deletes += deleted_count
                logger.info("Pattern invalidation successful", 
                           pattern=pattern, deleted_count=deleted_count)
                return deleted_count
            
            return 0
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Pattern invalidation failed", pattern=pattern, error=str(e))
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            redis_client = await get_redis()
            redis_info = await redis_client.info()
            
            return {
                "cache_stats": {
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "sets": self.stats.sets,
                    "deletes": self.stats.deletes,
                    "errors": self.stats.errors,
                    "hit_ratio": self.stats.hit_ratio
                },
                "redis_stats": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0)
                }
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"error": str(e)}


# Global cache service instance
cache_service = CacheService()


def cache_result(key_template: str = None, ttl: int = None, namespace: str = None):
    """Decorator to cache function results"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_template:
                cache_key = key_template.format(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            service = CacheService(namespace or "finops", ttl or 3600)
            cached_result = await service.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await service.set(cache_key, result, ttl)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'll need to handle differently
            # This is a simplified version - in production you might want async support
            result = func(*args, **kwargs)
            return result
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def invalidate_cache(patterns: List[str]):
    """Decorator to invalidate cache patterns after function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache patterns
            service = CacheService()
            for pattern in patterns:
                await service.invalidate_pattern(pattern)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Note: Sync version would need different handling for cache invalidation
            return result
        
        if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience functions for common cache operations
async def cache_cost_data(provider_id: str, start_date: str, end_date: str, data: Any, ttl: int = 1800):
    """Cache cost data with standardized key format"""
    key = f"cost_data:{provider_id}:{start_date}:{end_date}"
    return await cache_service.set(key, data, ttl)


async def get_cached_cost_data(provider_id: str, start_date: str, end_date: str) -> Optional[Any]:
    """Get cached cost data"""
    key = f"cost_data:{provider_id}:{start_date}:{end_date}"
    return await cache_service.get(key)


async def cache_budget_data(budget_id: str, data: Any, ttl: int = 900):
    """Cache budget data with standardized key format"""
    key = f"budget:{budget_id}"
    return await cache_service.set(key, data, ttl)


async def get_cached_budget_data(budget_id: str) -> Optional[Any]:
    """Get cached budget data"""
    key = f"budget:{budget_id}"
    return await cache_service.get(key)


async def invalidate_cost_data_cache(provider_id: str = None):
    """Invalidate cost data cache for a provider or all providers"""
    pattern = f"cost_data:{provider_id}:*" if provider_id else "cost_data:*"
    return await cache_service.invalidate_pattern(pattern)


async def invalidate_budget_cache(budget_id: str = None):
    """Invalidate budget cache for a specific budget or all budgets"""
    pattern = f"budget:{budget_id}" if budget_id else "budget:*"
    return await cache_service.invalidate_pattern(pattern)