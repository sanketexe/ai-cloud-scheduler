"""
Cache Integration Example

This module demonstrates how to integrate the caching system with existing
FinOps platform endpoints and services.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from .cache_service import cache_service, cache_result
from .performance_monitor import performance_monitor
from .cache_invalidation import invalidation_engine

router = APIRouter(prefix="/examples", tags=["cache-examples"])


# Example 1: Using cache decorators
@cache_result(key_template="cost_summary:{0}:{1}", ttl=1800)  # 30 minutes
async def get_cost_summary_cached(provider_id: str, date_range: str) -> Dict[str, Any]:
    """Example of cached cost summary with decorator"""
    
    # Simulate expensive operation
    async with performance_monitor.time_operation("cost_summary_calculation"):
        # This would normally query the database or external APIs
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "provider_id": provider_id,
            "date_range": date_range,
            "total_cost": 1234.56,
            "cost_by_service": {
                "EC2": 800.00,
                "S3": 200.00,
                "RDS": 234.56
            },
            "generated_at": datetime.utcnow().isoformat()
        }


# Example 2: Manual cache usage
@router.get("/cost-summary/{provider_id}")
async def get_cost_summary_manual(provider_id: str, date_range: str = "30d"):
    """Example of manual cache usage"""
    
    # Try to get from cache first
    cache_key = f"cost_summary:{provider_id}:{date_range}"
    cached_result = await cache_service.get(cache_key)
    
    if cached_result:
        performance_monitor.record_cache_hit_rate(cache_key, hit=True)
        return {
            "data": cached_result,
            "cached": True,
            "cache_key": cache_key
        }
    
    # Cache miss - calculate and cache result
    performance_monitor.record_cache_hit_rate(cache_key, hit=False)
    
    async with performance_monitor.time_operation("cost_summary_generation"):
        # Simulate expensive calculation
        result = {
            "provider_id": provider_id,
            "date_range": date_range,
            "total_cost": 1234.56,
            "cost_by_service": {
                "EC2": 800.00,
                "S3": 200.00,
                "RDS": 234.56
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Cache the result for 30 minutes
        await cache_service.set(cache_key, result, ttl=1800)
        
        # Tag the cache entry for easy invalidation
        invalidation_engine.tag_manager.add_key_to_tag(cache_key, "cost_data")
        
        return {
            "data": result,
            "cached": False,
            "cache_key": cache_key
        }


# Example 3: Cache invalidation on data updates
@router.post("/cost-data/{provider_id}/refresh")
async def refresh_cost_data(provider_id: str):
    """Example of cache invalidation when data is updated"""
    
    try:
        # Simulate data refresh operation
        async with performance_monitor.time_operation("cost_data_refresh"):
            # This would normally sync with cloud provider APIs
            await asyncio.sleep(0.5)  # Simulate API calls
        
        # Invalidate related cache entries
        pattern = f"cost_summary:{provider_id}:*"
        deleted_count = await invalidation_engine.invalidate_by_pattern(pattern)
        
        # Also invalidate by tag
        tag_deleted_count = await invalidation_engine.invalidate_by_tag("cost_data")
        
        # Trigger event-driven invalidation
        await invalidation_engine.invalidate_by_event("cost_data_updated", {
            "provider_id": provider_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "message": "Cost data refreshed and cache invalidated",
            "invalidated_entries": {
                "pattern_based": deleted_count,
                "tag_based": tag_deleted_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh cost data: {str(e)}")


# Example 4: Performance monitoring integration
@router.get("/dashboard/{user_id}")
async def get_dashboard_data(user_id: str):
    """Example of dashboard endpoint with performance monitoring"""
    
    # Monitor the entire operation
    async with performance_monitor.time_operation("dashboard_generation", 
                                                 tags={"user_id": user_id}):
        
        # Check cache first
        cache_key = f"dashboard:{user_id}"
        cached_data = await cache_service.get(cache_key)
        
        if cached_data:
            performance_monitor.record_cache_hit_rate(cache_key, hit=True)
            return {
                "data": cached_data,
                "cached": True,
                "generated_at": cached_data.get("generated_at")
            }
        
        # Generate dashboard data
        performance_monitor.record_cache_hit_rate(cache_key, hit=False)
        
        # Simulate multiple data sources
        async with performance_monitor.time_operation("cost_data_fetch"):
            cost_data = {"total_cost": 5000.00, "trend": "+5%"}
        
        async with performance_monitor.time_operation("budget_data_fetch"):
            budget_data = {"total_budget": 10000.00, "utilization": 0.5}
        
        async with performance_monitor.time_operation("alerts_fetch"):
            alerts_data = {"active_alerts": 2, "critical_alerts": 0}
        
        # Combine data
        dashboard_data = {
            "user_id": user_id,
            "cost_overview": cost_data,
            "budget_overview": budget_data,
            "alerts": alerts_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Cache for 15 minutes
        await cache_service.set(cache_key, dashboard_data, ttl=900)
        
        # Tag for invalidation
        invalidation_engine.tag_manager.add_key_to_tag(cache_key, "dashboard_data")
        
        return {
            "data": dashboard_data,
            "cached": False,
            "generated_at": dashboard_data["generated_at"]
        }


# Example 5: Batch cache operations
@router.get("/reports/batch")
async def get_batch_reports():
    """Example of batch cache operations"""
    
    report_types = ["cost", "budget", "optimization", "compliance"]
    results = {}
    
    async with performance_monitor.time_operation("batch_reports_generation"):
        
        for report_type in report_types:
            cache_key = f"report:{report_type}:summary"
            
            # Try to get from cache
            cached_report = await cache_service.get(cache_key)
            
            if cached_report:
                results[report_type] = {
                    "data": cached_report,
                    "cached": True
                }
                performance_monitor.record_cache_hit_rate(cache_key, hit=True)
            else:
                # Generate report
                performance_monitor.record_cache_hit_rate(cache_key, hit=False)
                
                # Simulate report generation
                report_data = {
                    "type": report_type,
                    "summary": f"Summary for {report_type} report",
                    "generated_at": datetime.utcnow().isoformat()
                }
                
                # Cache for 1 hour
                await cache_service.set(cache_key, report_data, ttl=3600)
                
                # Tag for invalidation
                invalidation_engine.tag_manager.add_key_to_tag(cache_key, "report_data")
                
                results[report_type] = {
                    "data": report_data,
                    "cached": False
                }
    
    return {
        "reports": results,
        "total_reports": len(report_types),
        "cache_summary": await cache_service.get_stats()
    }


# Example 6: Cache warming endpoint
@router.post("/cache/warm-common-data")
async def warm_common_cache():
    """Example of cache warming for frequently accessed data"""
    
    warming_functions = {
        "global_cost_summary": _generate_global_cost_summary,
        "top_services": _generate_top_services,
        "budget_alerts": _generate_budget_alerts,
        "optimization_opportunities": _generate_optimization_opportunities
    }
    
    try:
        await invalidation_engine.warm_cache(warming_functions)
        
        return {
            "status": "success",
            "message": "Common cache data warmed successfully",
            "warmed_keys": list(warming_functions.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache warming failed: {str(e)}")


# Helper functions for cache warming
async def _generate_global_cost_summary():
    """Generate global cost summary for cache warming"""
    return {
        "total_cost": 50000.00,
        "monthly_trend": "+3.2%",
        "top_cost_centers": ["Production", "Development", "Staging"],
        "generated_at": datetime.utcnow().isoformat()
    }


async def _generate_top_services():
    """Generate top services data for cache warming"""
    return {
        "services": [
            {"name": "EC2", "cost": 20000.00, "percentage": 40},
            {"name": "S3", "cost": 10000.00, "percentage": 20},
            {"name": "RDS", "cost": 8000.00, "percentage": 16}
        ],
        "generated_at": datetime.utcnow().isoformat()
    }


async def _generate_budget_alerts():
    """Generate budget alerts data for cache warming"""
    return {
        "total_alerts": 5,
        "critical_alerts": 1,
        "warning_alerts": 4,
        "alerts": [
            {"budget_name": "Production EC2", "utilization": 0.95, "severity": "critical"},
            {"budget_name": "Development S3", "utilization": 0.85, "severity": "warning"}
        ],
        "generated_at": datetime.utcnow().isoformat()
    }


async def _generate_optimization_opportunities():
    """Generate optimization opportunities for cache warming"""
    return {
        "total_opportunities": 12,
        "potential_savings": 5000.00,
        "top_opportunities": [
            {"type": "rightsizing", "savings": 2000.00, "resources": 15},
            {"type": "reserved_instances", "savings": 1800.00, "resources": 8},
            {"type": "unused_resources", "savings": 1200.00, "resources": 25}
        ],
        "generated_at": datetime.utcnow().isoformat()
    }


import asyncio