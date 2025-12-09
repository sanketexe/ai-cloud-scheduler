"""
FinOps Platform - Main API Entry Point with Phase 1 Foundation
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import structlog

# Import our core modules
from backend.core.database import db_config, database_health_check
from backend.core.auth_endpoints import auth_router
from backend.core.models import Base

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting FinOps Platform API")
    
    try:
        # Initialize database
        logger.info("Initializing database connection")
        await db_config.check_connection()
        
        # Create tables if they don't exist
        # In production, this should be handled by migrations
        if os.getenv("AUTO_CREATE_TABLES", "false").lower() == "true":
            logger.info("Creating database tables")
            db_config.create_tables()
        
        # Initialize Redis connection
        logger.info("Initializing Redis connection")
        from backend.core.redis_config import redis_manager
        redis_health = await redis_manager.health_check()
        if redis_health["status"] == "healthy":
            logger.info("Redis connection established successfully")
        else:
            logger.warning("Redis connection issues detected", **redis_health)
        
        logger.info("FinOps Platform API started successfully")
        
    except Exception as e:
        logger.error("Failed to start FinOps Platform API", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FinOps Platform API")
    
    # Close database connections
    db_config.close_connections()
    
    # Close Redis connections
    try:
        from backend.core.redis_config import redis_manager
        await redis_manager.close_connections()
        logger.info("Redis connections closed")
    except Exception as e:
        logger.error("Error closing Redis connections", error=str(e))
    
    logger.info("FinOps Platform API shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="FinOps Platform API",
    description="Enterprise Cloud Financial Operations Platform with real-time cost management, optimization, and governance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0").split(",")
)

# CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()
    
    # Generate correlation ID
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    
    # Add correlation ID to request state
    request.state.correlation_id = correlation_id
    
    # Log request
    logger.info(
        "HTTP request started",
        method=request.method,
        url=str(request.url),
        correlation_id=correlation_id,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            "HTTP request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration,
            correlation_id=correlation_id
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "HTTP request failed",
            method=request.method,
            url=str(request.url),
            duration=duration,
            correlation_id=correlation_id,
            error=str(e)
        )
        raise

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.error(
        "Unhandled exception",
        error=str(exc),
        correlation_id=correlation_id,
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        correlation_id=correlation_id,
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# Import additional routers
from backend.core.cloud_endpoints import cloud_router
# from backend.core.task_endpoints import router as task_router  # TODO: Fix import errors
from backend.core.cache_health_endpoints import router as cache_health_router
from backend.core.health_endpoints import router as health_router
from backend.core.ai_assistant_endpoints import router as ai_assistant_router
from backend.core.migration_advisor.assessment_endpoints import router as assessment_router
from backend.core.migration_advisor.requirements_endpoints import router as requirements_router
from backend.core.migration_advisor.recommendation_endpoints import router as recommendation_router
from backend.core.migration_advisor.migration_planning_endpoints import router as planning_router
from backend.core.migration_advisor.resource_organization_endpoints import router as resource_org_router
from backend.core.migration_advisor.dimensional_management_endpoints import router as dimensional_router
from backend.core.migration_advisor.integration_endpoints import router as integration_router

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(cloud_router, prefix="/api/v1")
# app.include_router(task_router)  # TODO: Fix import errors
app.include_router(cache_health_router, prefix="/api/v1")
app.include_router(health_router)
app.include_router(ai_assistant_router, prefix="/api/v1")
app.include_router(assessment_router, prefix="/api/v1")
app.include_router(requirements_router, prefix="/api/v1")
app.include_router(recommendation_router, prefix="/api/v1")
app.include_router(planning_router, prefix="/api/v1")
app.include_router(resource_org_router, prefix="/api/v1")
app.include_router(dimensional_router, prefix="/api/v1")
app.include_router(integration_router, prefix="/api/v1")

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FinOps Platform API",
        "version": "1.0.0",
        "description": "Enterprise Cloud Financial Operations Platform",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Check database
        db_health = await database_health_check()
        
        # Check Redis
        from backend.core.redis_config import redis_manager
        redis_health = await redis_manager.health_check()
        
        # Overall health
        overall_status = "healthy"
        if db_health["status"] != "healthy" or redis_health["status"] != "healthy":
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "services": {
                "database": db_health,
                "redis": redis_health
            },
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (placeholder)"""
    # This would integrate with prometheus_client
    return {"message": "Metrics endpoint - to be implemented with Prometheus"}

# Add missing imports
import time
import uuid
from datetime import datetime

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )