"""
API Logging Endpoints for Troubleshooting
Validates: Requirements 4.5
"""

from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from .api_call_logger import api_call_logger

router = APIRouter(prefix="/api/v1/logs", tags=["API Logging"])


class APILogStatistics(BaseModel):
    """Statistics about API calls"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    average_duration_ms: float
    total_retries: int
    average_retries_per_call: float


@router.get("/api-calls", summary="Get API call logs")
async def get_api_call_logs(
    service: Optional[str] = Query(None, description="Filter by service name"),
    operation: Optional[str] = Query(None, description="Filter by operation name"),
    failed_only: bool = Query(False, description="Only return failed calls"),
    limit: Optional[int] = Query(100, description="Maximum number of logs to return")
):
    """
    Get API call logs for troubleshooting
    
    Returns recent API calls with timing, errors, and retry information.
    Useful for debugging AWS integration issues during demos.
    """
    logs = api_call_logger.get_logs(
        service=service,
        operation=operation,
        failed_only=failed_only,
        limit=limit
    )
    
    return {
        'logs': logs,
        'count': len(logs)
    }


@router.get("/api-calls/statistics", response_model=APILogStatistics)
async def get_api_call_statistics():
    """
    Get API call statistics
    
    Returns aggregated statistics about API calls including
    success rates, average duration, and retry counts.
    """
    stats = api_call_logger.get_statistics()
    return APILogStatistics(**stats)


@router.get("/api-calls/errors")
async def get_error_summary():
    """
    Get summary of API errors
    
    Returns a breakdown of errors by error code to help
    identify common issues.
    """
    error_summary = api_call_logger.get_error_summary()
    
    return {
        'error_summary': error_summary,
        'total_unique_errors': len(error_summary),
        'total_errors': sum(error_summary.values())
    }


@router.delete("/api-calls", summary="Clear API call logs")
async def clear_api_call_logs():
    """
    Clear all API call logs
    
    Useful for starting fresh before a demo or presentation.
    """
    api_call_logger.clear_logs()
    
    return {
        'message': 'API call logs cleared successfully'
    }


@router.get("/health")
async def health_check():
    """Health check for API logging service"""
    stats = api_call_logger.get_statistics()
    
    return {
        'status': 'healthy',
        'service': 'API Logging',
        'total_logs': stats['total_calls']
    }
