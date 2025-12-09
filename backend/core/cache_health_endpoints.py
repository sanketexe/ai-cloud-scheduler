"""
Cache and Performance Health Check Endpoints

This module provides FastAPI endpoints for monitoring cache health,
performance metrics, and system status.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import structlog
from .redis_config import redis_manager
from .cache_service import cache_service
from .performance_monitor import performance_monitor
from .cache_invalidation import invalidation_engine

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/cache")
async def get_cache_health() -> Dict[str, Any]:
    """Get Redis cache health status"""
    try:
        health_status = await redis_manager.health_check()
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "cache_health": health_status,
            "cache_stats": await cache_service.get_stats()
        }
    except Exception as e:
        logger.error("Cache health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Cache health check failed: {str(e)}")


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive performance metrics"""
    try:
        return await performance_monitor.get_performance_summary()
    except Exception as e:
        logger.error("Performance metrics retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance metrics unavailable: {str(e)}")


@router.get("/cache/stats")
async def get_cache_statistics() -> Dict[str, Any]:
    """Get detailed cache statistics"""
    try:
        stats = await cache_service.get_stats()
        return {
            "cache_service_stats": stats,
            "performance_cache_metrics": performance_monitor.collector.get_cache_metrics_summary()
        }
    except Exception as e:
        logger.error("Cache statistics retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache statistics unavailable: {str(e)}")


@router.post("/cache/invalidate/pattern")
async def invalidate_cache_pattern(pattern: str) -> Dict[str, Any]:
    """Invalidate cache entries matching a pattern"""
    try:
        deleted_count = await invalidation_engine.invalidate_by_pattern(pattern)
        return {
            "status": "success",
            "pattern": pattern,
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error("Cache pattern invalidation failed", pattern=pattern, error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {str(e)}")


@router.post("/cache/invalidate/tag")
async def invalidate_cache_tag(tag_name: str, cascade: bool = True) -> Dict[str, Any]:
    """Invalidate cache entries by tag"""
    try:
        deleted_count = await invalidation_engine.invalidate_by_tag(tag_name, cascade)
        return {
            "status": "success",
            "tag_name": tag_name,
            "cascade": cascade,
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error("Cache tag invalidation failed", tag_name=tag_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache tag invalidation failed: {str(e)}")


@router.post("/cache/warm")
async def warm_cache() -> Dict[str, Any]:
    """Warm cache with frequently accessed data"""
    try:
        # Define cache warming functions for common data
        warming_functions = {
            "dashboard_summary": _warm_dashboard_data,
            "cost_overview": _warm_cost_overview,
            "budget_summary": _warm_budget_summary
        }
        
        await invalidation_engine.warm_cache(warming_functions)
        
        return {
            "status": "success",
            "message": "Cache warming completed",
            "warmed_keys": list(warming_functions.keys())
        }
    except Exception as e:
        logger.error("Cache warming failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache warming failed: {str(e)}")


@router.get("/system")
async def get_system_health() -> Dict[str, Any]:
    """Get overall system health including cache and performance"""
    try:
        # Get cache health
        cache_health = await redis_manager.health_check()
        
        # Get performance summary
        performance_summary = await performance_monitor.get_performance_summary()
        
        # Determine overall health status
        overall_status = "healthy"
        if cache_health["status"] != "healthy":
            overall_status = "degraded"
        
        # Check performance thresholds
        api_metrics = performance_summary.get("api_metrics", {})
        for endpoint, metrics in api_metrics.items():
            if metrics.get("average_response_time", 0) > 5.0:  # 5 second threshold
                overall_status = "degraded"
                break
        
        cache_metrics = performance_summary.get("cache_metrics", {})
        if cache_metrics.get("hit_ratio", 1.0) < 0.8:  # 80% hit ratio threshold
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": cache_health.get("timestamp", "unknown"),
            "components": {
                "cache": {
                    "status": cache_health["status"],
                    "response_time_ms": cache_health.get("response_time_ms", 0),
                    "hit_ratio": cache_health.get("hit_ratio", 0)
                },
                "performance": {
                    "status": "healthy" if overall_status != "unhealthy" else "degraded",
                    "api_endpoints": len(api_metrics),
                    "average_cache_hit_ratio": cache_metrics.get("hit_ratio", 0)
                }
            },
            "details": {
                "cache_health": cache_health,
                "performance_summary": performance_summary
            }
        }
    except Exception as e:
        logger.error("System health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "unknown"
        }


@router.post("/performance/reset")
async def reset_performance_metrics() -> Dict[str, Any]:
    """Reset performance metrics (useful for testing)"""
    try:
        performance_monitor.reset_metrics()
        return {
            "status": "success",
            "message": "Performance metrics reset successfully"
        }
    except Exception as e:
        logger.error("Performance metrics reset failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance metrics reset failed: {str(e)}")


# Cache warming helper functions
async def _warm_dashboard_data():
    """Warm dashboard data cache"""
    # This would typically fetch and return dashboard summary data
    # For now, return mock data structure
    return {
        "total_cost": 0.0,
        "monthly_trend": 0.0,
        "top_services": [],
        "budget_utilization": 0.0,
        "last_updated": "2024-01-01T00:00:00Z"
    }


async def _warm_cost_overview():
    """Warm cost overview cache"""
    # This would typically fetch and return cost overview data
    return {
        "current_month_cost": 0.0,
        "previous_month_cost": 0.0,
        "cost_by_service": {},
        "cost_by_region": {},
        "last_updated": "2024-01-01T00:00:00Z"
    }


async def _warm_budget_summary():
    """Warm budget summary cache"""
    # This would typically fetch and return budget summary data
    return {
        "total_budgets": 0,
        "active_budgets": 0,
        "budget_alerts": 0,
        "total_budget_amount": 0.0,
        "total_spent": 0.0,
        "last_updated": "2024-01-01T00:00:00Z"
    }