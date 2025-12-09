"""
Performance Monitoring System

This module provides comprehensive performance monitoring including API response times,
cache hit/miss ratios, database query performance, and system metrics for the FinOps platform.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
import threading
import structlog
from .redis_config import get_redis
from .enums import MetricType

logger = structlog.get_logger(__name__)


# Alias for backward compatibility
MetricsData = Dict[str, Any]


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIMetrics:
    """API endpoint performance metrics"""
    endpoint: str
    method: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def average_response_time(self) -> float:
        return self.total_response_time / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else 0.0


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_sets: int = 0
    cache_deletes: int = 0
    cache_errors: int = 0
    total_response_time: float = 0.0
    
    @property
    def hit_ratio(self) -> float:
        total_gets = self.cache_hits + self.cache_misses
        return self.cache_hits / total_gets if total_gets > 0 else 0.0
    
    @property
    def average_response_time(self) -> float:
        return self.total_response_time / self.total_requests if self.total_requests > 0 else 0.0


@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_query_time: float = 0.0
    slow_queries: int = 0
    slow_query_threshold: float = 1.0  # seconds
    query_types: Dict[str, int] = field(default_factory=dict)
    
    @property
    def average_query_time(self) -> float:
        return self.total_query_time / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0


class PerformanceCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_metrics_history: int = 10000):
        self.max_metrics_history = max_metrics_history
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.api_metrics: Dict[str, APIMetrics] = {}
        self.cache_metrics = CacheMetrics()
        self.database_metrics = DatabaseMetrics()
        self.custom_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a custom performance metric"""
        with self._lock:
            self.metrics_history.append(metric)
            self.custom_metrics[metric.name].append(metric)
            
            # Keep only recent metrics for each type
            if len(self.custom_metrics[metric.name]) > 1000:
                self.custom_metrics[metric.name] = self.custom_metrics[metric.name][-1000:]
        
        logger.debug("Performance metric recorded", 
                    name=metric.name, value=metric.value, unit=metric.unit)
    
    def record_api_call(self, endpoint: str, method: str, status_code: int, 
                       response_time: float):
        """Record API call performance metrics"""
        with self._lock:
            key = f"{method}:{endpoint}"
            
            if key not in self.api_metrics:
                self.api_metrics[key] = APIMetrics(endpoint=endpoint, method=method)
            
            metrics = self.api_metrics[key]
            metrics.total_requests += 1
            metrics.total_response_time += response_time
            metrics.response_times.append(response_time)
            
            if response_time < metrics.min_response_time:
                metrics.min_response_time = response_time
            if response_time > metrics.max_response_time:
                metrics.max_response_time = response_time
            
            if 200 <= status_code < 400:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
        
        logger.debug("API call recorded", 
                    endpoint=endpoint, method=method, 
                    status_code=status_code, response_time=response_time)
    
    def record_cache_operation(self, operation: str, hit: bool = None, 
                             response_time: float = 0.0, error: bool = False):
        """Record cache operation metrics"""
        with self._lock:
            self.cache_metrics.total_requests += 1
            self.cache_metrics.total_response_time += response_time
            
            if error:
                self.cache_metrics.cache_errors += 1
            elif operation == "get":
                if hit:
                    self.cache_metrics.cache_hits += 1
                else:
                    self.cache_metrics.cache_misses += 1
            elif operation == "set":
                self.cache_metrics.cache_sets += 1
            elif operation == "delete":
                self.cache_metrics.cache_deletes += 1
        
        logger.debug("Cache operation recorded", 
                    operation=operation, hit=hit, response_time=response_time)
    
    def record_database_query(self, query_type: str, execution_time: float, 
                            success: bool = True):
        """Record database query performance metrics"""
        with self._lock:
            self.database_metrics.total_queries += 1
            self.database_metrics.total_query_time += execution_time
            
            if success:
                self.database_metrics.successful_queries += 1
            else:
                self.database_metrics.failed_queries += 1
            
            if execution_time > self.database_metrics.slow_query_threshold:
                self.database_metrics.slow_queries += 1
            
            if query_type not in self.database_metrics.query_types:
                self.database_metrics.query_types[query_type] = 0
            self.database_metrics.query_types[query_type] += 1
        
        logger.debug("Database query recorded", 
                    query_type=query_type, execution_time=execution_time, success=success)
    
    def get_api_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of API performance metrics"""
        with self._lock:
            summary = {}
            for key, metrics in self.api_metrics.items():
                summary[key] = {
                    "total_requests": metrics.total_requests,
                    "success_rate": metrics.success_rate,
                    "average_response_time": metrics.average_response_time,
                    "min_response_time": metrics.min_response_time,
                    "max_response_time": metrics.max_response_time,
                    "p95_response_time": metrics.p95_response_time
                }
            return summary
    
    def get_cache_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of cache performance metrics"""
        with self._lock:
            return {
                "total_requests": self.cache_metrics.total_requests,
                "hit_ratio": self.cache_metrics.hit_ratio,
                "cache_hits": self.cache_metrics.cache_hits,
                "cache_misses": self.cache_metrics.cache_misses,
                "cache_sets": self.cache_metrics.cache_sets,
                "cache_deletes": self.cache_metrics.cache_deletes,
                "cache_errors": self.cache_metrics.cache_errors,
                "average_response_time": self.cache_metrics.average_response_time
            }
    
    def get_database_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of database performance metrics"""
        with self._lock:
            return {
                "total_queries": self.database_metrics.total_queries,
                "success_rate": self.database_metrics.success_rate,
                "average_query_time": self.database_metrics.average_query_time,
                "slow_queries": self.database_metrics.slow_queries,
                "slow_query_threshold": self.database_metrics.slow_query_threshold,
                "query_types": dict(self.database_metrics.query_types)
            }


class PerformanceMonitor:
    """Main performance monitoring class with context managers and decorators"""
    
    def __init__(self):
        self.collector = PerformanceCollector()
        self.alert_thresholds = {
            "api_response_time": 5.0,  # seconds
            "cache_hit_ratio": 0.8,    # 80%
            "database_query_time": 2.0, # seconds
            "error_rate": 0.05         # 5%
        }
        self.alert_callbacks: List[Callable] = []
    
    @asynccontextmanager
    async def time_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager to time async operations"""
        start_time = time.time()
        tags = tags or {}
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            tags["error"] = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            metric = PerformanceMetric(
                name=operation_name,
                value=duration,
                unit="seconds",
                timestamp=datetime.utcnow(),
                tags=tags,
                metadata={"success": success}
            )
            
            self.collector.record_metric(metric)
    
    @contextmanager
    def time_sync_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager to time synchronous operations"""
        start_time = time.time()
        tags = tags or {}
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            tags["error"] = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            metric = PerformanceMetric(
                name=operation_name,
                value=duration,
                unit="seconds",
                timestamp=datetime.utcnow(),
                tags=tags,
                metadata={"success": success}
            )
            
            self.collector.record_metric(metric)
    
    def time_function(self, operation_name: str = None):
        """Decorator to time function execution"""
        def decorator(func: Callable) -> Callable:
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with self.time_operation(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    with self.time_sync_operation(name):
                        return func(*args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    def record_api_call(self, endpoint: str, method: str, status_code: int, 
                       response_time: float):
        """Record API call metrics and check for alerts"""
        self.collector.record_api_call(endpoint, method, status_code, response_time)
        
        # Check for performance alerts
        if response_time > self.alert_thresholds["api_response_time"]:
            self._trigger_alert("api_response_time", {
                "endpoint": endpoint,
                "method": method,
                "response_time": response_time,
                "threshold": self.alert_thresholds["api_response_time"]
            })
    
    def record_cache_hit_rate(self, cache_key: str, hit: bool, response_time: float = 0.0):
        """Record cache hit/miss and check hit ratio"""
        operation = "get"
        self.collector.record_cache_operation(operation, hit, response_time)
        
        # Check cache hit ratio periodically
        if self.collector.cache_metrics.total_requests % 100 == 0:  # Check every 100 requests
            hit_ratio = self.collector.cache_metrics.hit_ratio
            if hit_ratio < self.alert_thresholds["cache_hit_ratio"]:
                self._trigger_alert("cache_hit_ratio", {
                    "current_ratio": hit_ratio,
                    "threshold": self.alert_thresholds["cache_hit_ratio"],
                    "total_requests": self.collector.cache_metrics.total_requests
                })
    
    def record_database_query(self, query_type: str, execution_time: float, 
                            success: bool = True):
        """Record database query metrics and check for slow queries"""
        self.collector.record_database_query(query_type, execution_time, success)
        
        # Check for slow query alert
        if execution_time > self.alert_thresholds["database_query_time"]:
            self._trigger_alert("database_slow_query", {
                "query_type": query_type,
                "execution_time": execution_time,
                "threshold": self.alert_thresholds["database_query_time"]
            })
    
    def record_external_api_call(self, provider: str, endpoint: str, 
                               duration: float, success: bool):
        """Record external API call metrics"""
        tags = {"provider": provider, "endpoint": endpoint}
        metric = PerformanceMetric(
            name="external_api_call",
            value=duration,
            unit="seconds",
            timestamp=datetime.utcnow(),
            tags=tags,
            metadata={"success": success}
        )
        
        self.collector.record_metric(metric)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger performance alert"""
        alert_data = {
            "type": alert_type,
            "timestamp": datetime.utcnow(),
            "data": data
        }
        
        logger.warning("Performance alert triggered", **alert_data)
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Get Redis performance info
            redis_client = await get_redis()
            redis_info = await redis_client.info()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "api_metrics": self.collector.get_api_metrics_summary(),
                "cache_metrics": self.collector.get_cache_metrics_summary(),
                "database_metrics": self.collector.get_database_metrics_summary(),
                "redis_info": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0)
                },
                "alert_thresholds": self.alert_thresholds
            }
        except Exception as e:
            logger.error("Failed to get performance summary", error=str(e))
            return {"error": str(e)}
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.collector = PerformanceCollector()
        logger.info("Performance metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Convenience functions for common monitoring scenarios
async def monitor_api_endpoint(endpoint: str, method: str):
    """Context manager for monitoring API endpoint performance"""
    return performance_monitor.time_operation(f"api_{method}_{endpoint}")


def monitor_database_operation(operation_type: str):
    """Context manager for monitoring database operations"""
    return performance_monitor.time_sync_operation(f"db_{operation_type}")


def monitor_cache_operation(operation_type: str):
    """Context manager for monitoring cache operations"""
    return performance_monitor.time_sync_operation(f"cache_{operation_type}")


def monitor_external_api(provider: str, endpoint: str):
    """Context manager for monitoring external API calls"""
    return performance_monitor.time_sync_operation(
        f"external_api_{provider}_{endpoint}",
        tags={"provider": provider, "endpoint": endpoint}
    )