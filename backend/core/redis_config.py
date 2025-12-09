"""
Redis Configuration and Connection Management

This module provides Redis configuration, connection pooling, and health checks
for the FinOps platform caching system.
"""

import os
import redis
import redis.asyncio as aioredis
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import structlog
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)


class RedisConfig:
    """Redis configuration with connection pooling and health checks"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        self.socket_connect_timeout = int(os.getenv("REDIS_CONNECT_TIMEOUT", "5"))
        self.retry_on_timeout = True
        self.health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
        
        # Parse Redis URL for connection details
        parsed_url = urlparse(self.redis_url)
        self.host = parsed_url.hostname or "localhost"
        self.port = parsed_url.port or 6379
        self.password = parsed_url.password
        self.db = int(parsed_url.path.lstrip('/')) if parsed_url.path else 0
        
        # Connection pool settings
        self.connection_pool_kwargs = {
            "host": self.host,
            "port": self.port,
            "password": self.password,
            "db": self.db,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "retry_on_timeout": self.retry_on_timeout,
            "decode_responses": True,
            "encoding": "utf-8"
        }
        
        logger.info("Redis configuration initialized", 
                   host=self.host, port=self.port, db=self.db)


class RedisConnectionManager:
    """Manages Redis connections with connection pooling and health monitoring"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._sync_pool: Optional[redis.ConnectionPool] = None
        self._async_pool: Optional[aioredis.ConnectionPool] = None
        self._sync_client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
        
    def get_sync_client(self) -> redis.Redis:
        """Get synchronous Redis client with connection pooling"""
        if self._sync_client is None:
            if self._sync_pool is None:
                self._sync_pool = redis.ConnectionPool(**self.config.connection_pool_kwargs)
            
            self._sync_client = redis.Redis(connection_pool=self._sync_pool)
            logger.info("Synchronous Redis client created")
            
        return self._sync_client
    
    async def get_async_client(self) -> aioredis.Redis:
        """Get asynchronous Redis client with connection pooling"""
        if self._async_client is None:
            if self._async_pool is None:
                # Create async connection pool
                pool_kwargs = self.config.connection_pool_kwargs.copy()
                self._async_pool = aioredis.ConnectionPool(**pool_kwargs)
            
            self._async_client = aioredis.Redis(connection_pool=self._async_pool)
            logger.info("Asynchronous Redis client created")
            
        return self._async_client
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check and return status information"""
        try:
            import time
            client = await self.get_async_client()
            
            # Test basic connectivity
            start_time = time.time()
            ping_result = await client.ping()
            end_time = time.time()
            
            if not ping_result:
                raise redis.ConnectionError("Redis ping failed")
            
            # Get Redis info
            info = await client.info()
            
            # Calculate response time in milliseconds
            response_time = (end_time - start_time) * 1000
            
            health_status = {
                "status": "healthy",
                "response_time_ms": response_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
            
            # Calculate hit ratio
            hits = health_status["keyspace_hits"]
            misses = health_status["keyspace_misses"]
            total_requests = hits + misses
            
            if total_requests > 0:
                health_status["hit_ratio"] = hits / total_requests
            else:
                health_status["hit_ratio"] = 0.0
            
            logger.info("Redis health check successful", **health_status)
            return health_status
            
        except Exception as e:
            error_status = {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__
            }
            logger.error("Redis health check failed", **error_status)
            return error_status
    
    async def close_connections(self):
        """Close all Redis connections and clean up resources"""
        try:
            if self._async_client:
                await self._async_client.close()
                self._async_client = None
                logger.info("Async Redis client closed")
            
            if self._async_pool:
                await self._async_pool.disconnect()
                self._async_pool = None
                logger.info("Async Redis connection pool closed")
            
            if self._sync_client:
                self._sync_client.close()
                self._sync_client = None
                logger.info("Sync Redis client closed")
            
            if self._sync_pool:
                self._sync_pool.disconnect()
                self._sync_pool = None
                logger.info("Sync Redis connection pool closed")
                
        except Exception as e:
            logger.error("Error closing Redis connections", error=str(e))


# Global Redis connection manager instance
redis_config = RedisConfig()
redis_manager = RedisConnectionManager(redis_config)


@asynccontextmanager
async def get_redis_client():
    """Context manager for getting Redis client with automatic cleanup"""
    client = await redis_manager.get_async_client()
    try:
        yield client
    finally:
        # Connection is returned to pool automatically
        pass


async def get_redis() -> aioredis.Redis:
    """Dependency function for FastAPI to get Redis client"""
    return await redis_manager.get_async_client()


def get_sync_redis() -> redis.Redis:
    """Get synchronous Redis client for non-async contexts"""
    return redis_manager.get_sync_client()