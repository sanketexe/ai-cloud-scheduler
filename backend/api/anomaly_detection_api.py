"""
REST API Endpoints for AI-Powered Cost Anomaly Detection

Provides comprehensive API endpoints for anomaly detection configuration,
real-time anomaly queries, forecast generation, model performance monitoring,
and audit trail access.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from pathlib import Path

# FastAPI imports (would be actual imports in production)
try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Path as PathParam
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback for demo - create mock classes
    FASTAPI_AVAILABLE = False
    
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class FastAPI:
        def __init__(self, **kwargs):
            self.routes = []
        
        def get(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append(('GET', path, func))
                return func
            return decorator
        
        def post(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append(('POST', path, func))
                return func
            return decorator
        
        def put(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append(('PUT', path, func))
                return func
            return decorator
        
        def delete(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append(('DELETE', path, func))
                return func
            return decorator
    
    HTTPException = Exception
    JSONResponse = dict
    Query = lambda default=None, **kwargs: default
    PathParam = lambda **kwargs: None
    Field = lambda default=None, **kwargs: default

logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class AnomalyConfigurationRequest(BaseModel):
    """Request model for anomaly detection configuration"""
    sensitivity_level: str = Field(default="balanced", description="Sensitivity level: conservative, balanced, aggressive")
    threshold_percentage: float = Field(default=20.0, description="Anomaly threshold percentage")
    baseline_period_days: int = Field(default=30, description="Baseline period in days")
    min_cost_threshold: float = Field(default=1.0, description="Minimum cost threshold for detection")
    excluded_services: List[str] = Field(default=[], description="Services to exclude from detection")
    maintenance_windows: List[Dict[str, str]] = Field(default=[], description="Maintenance windows to exclude")
    notification_channels: List[str] = Field(default=["email"], description="Notification channels")
    escalation_rules: Dict[str, Any] = Field(default={}, description="Alert escalation rules")

class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    account_id: str = Field(description="AWS account ID")
    time_range: Dict[str, str] = Field(description="Time range for detection")
    services: Optional[List[str]] = Field(default=None, description="Specific services to analyze")
    regions: Optional[List[str]] = Field(default=None, description="Specific regions to analyze")
    cost_threshold: Optional[float] = Field(default=None, description="Minimum cost threshold")
    include_forecasts: bool = Field(default=False, description="Include forecast data")

class ForecastRequest(BaseModel):
    """Request model for cost forecasting"""
    account_id: str = Field(description="AWS account ID")
    forecast_horizon_days: int = Field(default=30, description="Forecast horizon in days")
    confidence_level: float = Field(default=0.8, description="Confidence level for intervals")
    include_seasonality: bool = Field(default=True, description="Include seasonal adjustments")
    services: Optional[List[str]] = Field(default=None, description="Specific services to forecast")
    granularity: str = Field(default="daily", description="Forecast granularity: daily, weekly, monthly")

class ModelPerformanceRequest(BaseModel):
    """Request model for model performance queries"""
    model_id: Optional[str] = Field(default=None, description="Specific model ID")
    time_range: Dict[str, str] = Field(description="Time range for performance data")
    metrics: Optional[List[str]] = Field(default=None, description="Specific metrics to retrieve")
    include_drift_analysis: bool = Field(default=False, description="Include drift analysis")

class AuditTrailRequest(BaseModel):
    """Request model for audit trail queries"""
    event_types: Optional[List[str]] = Field(default=None, description="Event types to filter")
    user_id: Optional[str] = Field(default=None, description="User ID to filter")
    resource_id: Optional[str] = Field(default=None, description="Resource ID to filter")
    time_range: Dict[str, str] = Field(description="Time range for audit events")
    limit: int = Field(default=100, description="Maximum number of events to return")

# Response models
class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection results"""
    anomalies: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]

class ForecastResponse(BaseModel):
    """Response model for forecast results"""
    forecasts: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Any]
    metadata: Dict[str, Any]

class ModelPerformanceResponse(BaseModel):
    """Response model for model performance"""
    performance_metrics: Dict[str, Any]
    drift_analysis: Optional[Dict[str, Any]]
    recommendations: List[str]

class AuditTrailResponse(BaseModel):
    """Response model for audit trail"""
    events: List[Dict[str, Any]]
    total_count: int
    metadata: Dict[str, Any]
class AnomalyDetectionAPI:
    """
    Comprehensive REST API for AI-Powered Cost Anomaly Detection.
    
    Provides endpoints for configuration, real-time queries, forecasting,
    model performance monitoring, and audit trail access.
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="AI-Powered Cost Anomaly Detection API",
            description="Comprehensive API for cost anomaly detection, forecasting, and monitoring",
            version="1.0.0"
        )
        
        # Add CORS middleware if FastAPI is available
        if FASTAPI_AVAILABLE:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Mock data storage (in production, this would connect to actual services)
        self.configurations = {}
        self.anomaly_cache = {}
        self.forecast_cache = {}
        self.performance_data = {}
        self.audit_events = []
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        # Configuration endpoints
        @self.app.get("/api/v1/config", response_model=dict)
        async def get_configuration(account_id: str = Query(..., description="AWS account ID")):
            """Get current anomaly detection configuration"""
            return await self.get_configuration(account_id)
        
        @self.app.post("/api/v1/config", response_model=dict)
        async def update_configuration(
            account_id: str = Query(..., description="AWS account ID"),
            config: AnomalyConfigurationRequest = None
        ):
            """Update anomaly detection configuration"""
            return await self.update_configuration(account_id, config)
        
        # Anomaly detection endpoints
        @self.app.post("/api/v1/anomalies/detect", response_model=AnomalyDetectionResponse)
        async def detect_anomalies(request: AnomalyDetectionRequest):
            """Detect cost anomalies for specified parameters"""
            return await self.detect_anomalies(request)
        
        @self.app.get("/api/v1/anomalies", response_model=dict)
        async def get_anomalies(
            account_id: str = Query(..., description="AWS account ID"),
            start_date: str = Query(..., description="Start date (ISO format)"),
            end_date: str = Query(..., description="End date (ISO format)"),
            severity: Optional[str] = Query(None, description="Severity filter"),
            limit: int = Query(100, description="Maximum results")
        ):
            """Get historical anomalies"""
            return await self.get_anomalies(account_id, start_date, end_date, severity, limit)
        
        @self.app.get("/api/v1/anomalies/{anomaly_id}", response_model=dict)
        async def get_anomaly_details(anomaly_id: str = PathParam(..., description="Anomaly ID")):
            """Get detailed information about a specific anomaly"""
            return await self.get_anomaly_details(anomaly_id)
        
        # Forecasting endpoints
        @self.app.post("/api/v1/forecasts/generate", response_model=ForecastResponse)
        async def generate_forecast(request: ForecastRequest):
            """Generate cost forecasts"""
            return await self.generate_forecast(request)
        
        @self.app.get("/api/v1/forecasts", response_model=dict)
        async def get_forecasts(
            account_id: str = Query(..., description="AWS account ID"),
            horizon_days: int = Query(30, description="Forecast horizon"),
            services: Optional[str] = Query(None, description="Comma-separated services")
        ):
            """Get existing forecasts"""
            return await self.get_forecasts(account_id, horizon_days, services)
        
        # Model performance endpoints
        @self.app.get("/api/v1/models/performance", response_model=ModelPerformanceResponse)
        async def get_model_performance(
            model_id: Optional[str] = Query(None, description="Specific model ID"),
            start_date: str = Query(..., description="Start date (ISO format)"),
            end_date: str = Query(..., description="End date (ISO format)")
        ):
            """Get model performance metrics"""
            return await self.get_model_performance(model_id, start_date, end_date)
        
        @self.app.get("/api/v1/models", response_model=dict)
        async def list_models():
            """List all available models"""
            return await self.list_models()
        
        @self.app.get("/api/v1/models/{model_id}/drift", response_model=dict)
        async def get_model_drift(model_id: str = PathParam(..., description="Model ID")):
            """Get model drift analysis"""
            return await self.get_model_drift(model_id)
        # Audit trail endpoints
        @self.app.get("/api/v1/audit", response_model=AuditTrailResponse)
        async def get_audit_trail(
            event_types: Optional[str] = Query(None, description="Comma-separated event types"),
            user_id: Optional[str] = Query(None, description="User ID filter"),
            start_date: str = Query(..., description="Start date (ISO format)"),
            end_date: str = Query(..., description="End date (ISO format)"),
            limit: int = Query(100, description="Maximum results")
        ):
            """Get audit trail events"""
            return await self.get_audit_trail(event_types, user_id, start_date, end_date, limit)
        
        # Alert management endpoints
        @self.app.get("/api/v1/alerts", response_model=dict)
        async def get_alerts(
            account_id: str = Query(..., description="AWS account ID"),
            status: Optional[str] = Query(None, description="Alert status filter"),
            severity: Optional[str] = Query(None, description="Severity filter"),
            limit: int = Query(100, description="Maximum results")
        ):
            """Get active alerts"""
            return await self.get_alerts(account_id, status, severity, limit)
        
        @self.app.post("/api/v1/alerts/{alert_id}/acknowledge", response_model=dict)
        async def acknowledge_alert(
            alert_id: str = PathParam(..., description="Alert ID"),
            user_id: str = Query(..., description="User acknowledging the alert"),
            notes: Optional[str] = Query(None, description="Acknowledgment notes")
        ):
            """Acknowledge an alert"""
            return await self.acknowledge_alert(alert_id, user_id, notes)
        
        @self.app.post("/api/v1/alerts/{alert_id}/snooze", response_model=dict)
        async def snooze_alert(
            alert_id: str = PathParam(..., description="Alert ID"),
            duration_minutes: int = Query(..., description="Snooze duration in minutes"),
            user_id: str = Query(..., description="User snoozing the alert")
        ):
            """Snooze an alert"""
            return await self.snooze_alert(alert_id, duration_minutes, user_id)
        
        # Health and status endpoints
        @self.app.get("/api/v1/health", response_model=dict)
        async def health_check():
            """API health check"""
            return await self.health_check()
        
        @self.app.get("/api/v1/status", response_model=dict)
        async def get_system_status():
            """Get system status and statistics"""
            return await self.get_system_status()
    
    # Implementation methods
    async def get_configuration(self, account_id: str) -> dict:
        """Get anomaly detection configuration for account"""
        
        config = self.configurations.get(account_id, {
            "sensitivity_level": "balanced",
            "threshold_percentage": 20.0,
            "baseline_period_days": 30,
            "min_cost_threshold": 1.0,
            "excluded_services": [],
            "maintenance_windows": [],
            "notification_channels": ["email"],
            "escalation_rules": {},
            "last_updated": datetime.now().isoformat(),
            "created_by": "system"
        })
        
        return {
            "account_id": account_id,
            "configuration": config,
            "status": "active"
        }
    
    async def update_configuration(self, account_id: str, config: AnomalyConfigurationRequest) -> dict:
        """Update anomaly detection configuration"""
        
        # Validate configuration
        if config.threshold_percentage < 1.0 or config.threshold_percentage > 100.0:
            raise HTTPException(status_code=400, detail="Threshold percentage must be between 1.0 and 100.0")
        
        if config.baseline_period_days < 7 or config.baseline_period_days > 365:
            raise HTTPException(status_code=400, detail="Baseline period must be between 7 and 365 days")
        
        # Update configuration
        updated_config = {
            "sensitivity_level": config.sensitivity_level,
            "threshold_percentage": config.threshold_percentage,
            "baseline_period_days": config.baseline_period_days,
            "min_cost_threshold": config.min_cost_threshold,
            "excluded_services": config.excluded_services,
            "maintenance_windows": config.maintenance_windows,
            "notification_channels": config.notification_channels,
            "escalation_rules": config.escalation_rules,
            "last_updated": datetime.now().isoformat(),
            "updated_by": "api_user"
        }
        
        self.configurations[account_id] = updated_config
        
        # Log configuration change
        self._log_audit_event(
            event_type="configuration_change",
            description=f"Updated anomaly detection configuration for account {account_id}",
            resource_id=account_id,
            metadata={"changes": updated_config}
        )
        
        return {
            "account_id": account_id,
            "configuration": updated_config,
            "status": "updated",
            "message": "Configuration updated successfully"
        }
    async def detect_anomalies(self, request: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
        """Detect cost anomalies based on request parameters"""
        
        # Parse time range
        start_date = datetime.fromisoformat(request.time_range["start_date"].replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.time_range["end_date"].replace('Z', '+00:00'))
        
        # Simulate anomaly detection (in production, this would call actual ML models)
        anomalies = self._simulate_anomaly_detection(
            account_id=request.account_id,
            start_date=start_date,
            end_date=end_date,
            services=request.services,
            regions=request.regions,
            cost_threshold=request.cost_threshold
        )
        
        # Generate summary
        summary = {
            "total_anomalies": len(anomalies),
            "high_severity": len([a for a in anomalies if a["severity"] == "high"]),
            "medium_severity": len([a for a in anomalies if a["severity"] == "medium"]),
            "low_severity": len([a for a in anomalies if a["severity"] == "low"]),
            "total_estimated_impact": sum(a["estimated_impact_usd"] for a in anomalies),
            "detection_time_ms": 245.7,
            "model_confidence": 0.87
        }
        
        # Add forecasts if requested
        forecasts = []
        if request.include_forecasts:
            forecasts = self._generate_forecast_data(request.account_id, 7)
        
        metadata = {
            "account_id": request.account_id,
            "detection_timestamp": datetime.now().isoformat(),
            "time_range": request.time_range,
            "services_analyzed": request.services or ["all"],
            "regions_analyzed": request.regions or ["all"],
            "forecasts_included": request.include_forecasts,
            "forecasts": forecasts if request.include_forecasts else None
        }
        
        # Log detection request
        self._log_audit_event(
            event_type="anomaly_detection",
            description=f"Anomaly detection performed for account {request.account_id}",
            resource_id=request.account_id,
            metadata={"anomalies_found": len(anomalies), "summary": summary}
        )
        
        return AnomalyDetectionResponse(
            anomalies=anomalies,
            summary=summary,
            metadata=metadata
        )
    
    async def get_anomalies(self, account_id: str, start_date: str, end_date: str, 
                           severity: Optional[str], limit: int) -> dict:
        """Get historical anomalies"""
        
        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Get cached anomalies or generate new ones
        cache_key = f"{account_id}_{start_date}_{end_date}"
        if cache_key not in self.anomaly_cache:
            self.anomaly_cache[cache_key] = self._simulate_anomaly_detection(
                account_id=account_id,
                start_date=start_dt,
                end_date=end_dt
            )
        
        anomalies = self.anomaly_cache[cache_key]
        
        # Apply severity filter
        if severity:
            anomalies = [a for a in anomalies if a["severity"] == severity]
        
        # Apply limit
        anomalies = anomalies[:limit]
        
        return {
            "anomalies": anomalies,
            "total_count": len(anomalies),
            "filters": {
                "account_id": account_id,
                "start_date": start_date,
                "end_date": end_date,
                "severity": severity,
                "limit": limit
            },
            "metadata": {
                "retrieved_at": datetime.now().isoformat(),
                "cache_hit": cache_key in self.anomaly_cache
            }
        }
    
    async def get_anomaly_details(self, anomaly_id: str) -> dict:
        """Get detailed information about a specific anomaly"""
        
        # Simulate detailed anomaly information
        anomaly_details = {
            "anomaly_id": anomaly_id,
            "detection_timestamp": datetime.now().isoformat(),
            "severity": "high",
            "confidence_score": 0.92,
            "anomaly_score": 0.87,
            "estimated_impact_usd": 1250.75,
            "affected_services": ["EC2", "S3", "RDS"],
            "affected_regions": ["us-east-1", "us-west-2"],
            "root_cause_analysis": {
                "primary_cause": "Unexpected EC2 instance scaling",
                "contributing_factors": [
                    "Increased CPU utilization",
                    "Memory usage spike",
                    "Network I/O anomaly"
                ],
                "affected_resources": [
                    "i-1234567890abcdef0",
                    "i-0987654321fedcba0"
                ]
            },
            "time_series_data": [
                {"timestamp": "2024-01-01T00:00:00Z", "cost": 125.50, "baseline": 98.20},
                {"timestamp": "2024-01-01T01:00:00Z", "cost": 145.75, "baseline": 102.30},
                {"timestamp": "2024-01-01T02:00:00Z", "cost": 189.25, "baseline": 105.10}
            ],
            "recommendations": [
                "Review EC2 auto-scaling policies",
                "Investigate unusual traffic patterns",
                "Consider rightsizing instances"
            ],
            "similar_anomalies": [
                {"anomaly_id": "anom_456", "similarity_score": 0.85, "date": "2023-12-15"},
                {"anomaly_id": "anom_789", "similarity_score": 0.78, "date": "2023-11-22"}
            ]
        }
        
        return anomaly_details
    async def generate_forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Generate cost forecasts"""
        
        # Generate forecast data
        forecasts = self._generate_forecast_data(
            account_id=request.account_id,
            horizon_days=request.forecast_horizon_days,
            confidence_level=request.confidence_level,
            services=request.services,
            granularity=request.granularity
        )
        
        # Calculate confidence intervals
        confidence_intervals = {
            "upper_bound": [f["predicted_cost"] * (1 + (1 - request.confidence_level) / 2) for f in forecasts],
            "lower_bound": [f["predicted_cost"] * (1 - (1 - request.confidence_level) / 2) for f in forecasts],
            "confidence_level": request.confidence_level
        }
        
        metadata = {
            "account_id": request.account_id,
            "forecast_generated_at": datetime.now().isoformat(),
            "horizon_days": request.forecast_horizon_days,
            "granularity": request.granularity,
            "services_included": request.services or ["all"],
            "seasonality_included": request.include_seasonality,
            "model_version": "prophet_v2.1.0",
            "forecast_accuracy_estimate": 0.89
        }
        
        # Log forecast generation
        self._log_audit_event(
            event_type="forecast_generation",
            description=f"Cost forecast generated for account {request.account_id}",
            resource_id=request.account_id,
            metadata={"horizon_days": request.forecast_horizon_days, "data_points": len(forecasts)}
        )
        
        return ForecastResponse(
            forecasts=forecasts,
            confidence_intervals=confidence_intervals,
            metadata=metadata
        )
    
    async def get_forecasts(self, account_id: str, horizon_days: int, services: Optional[str]) -> dict:
        """Get existing forecasts"""
        
        services_list = services.split(",") if services else None
        
        # Get cached forecasts or generate new ones
        cache_key = f"forecast_{account_id}_{horizon_days}_{services or 'all'}"
        if cache_key not in self.forecast_cache:
            self.forecast_cache[cache_key] = self._generate_forecast_data(
                account_id=account_id,
                horizon_days=horizon_days,
                services=services_list
            )
        
        forecasts = self.forecast_cache[cache_key]
        
        return {
            "forecasts": forecasts,
            "account_id": account_id,
            "horizon_days": horizon_days,
            "services": services_list or ["all"],
            "generated_at": datetime.now().isoformat(),
            "cache_hit": cache_key in self.forecast_cache
        }
    
    async def get_model_performance(self, model_id: Optional[str], start_date: str, end_date: str) -> ModelPerformanceResponse:
        """Get model performance metrics"""
        
        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Generate performance data
        performance_metrics = self._generate_performance_data(model_id, start_dt, end_dt)
        
        # Generate drift analysis
        drift_analysis = {
            "drift_detected": True,
            "drift_score": 0.34,
            "drift_type": "data_drift",
            "affected_features": ["cost_amount", "cpu_utilization"],
            "recommendation": "Consider retraining model with recent data"
        }
        
        recommendations = [
            "Model performance is within acceptable range",
            "Monitor drift score - approaching threshold",
            "Consider A/B testing with newer model version"
        ]
        
        return ModelPerformanceResponse(
            performance_metrics=performance_metrics,
            drift_analysis=drift_analysis,
            recommendations=recommendations
        )
    
    async def list_models(self) -> dict:
        """List all available models"""
        
        models = [
            {
                "model_id": "isolation_forest_v1.0.0",
                "model_name": "Isolation Forest Anomaly Detector",
                "model_type": "isolation_forest",
                "status": "deployed",
                "environment": "production",
                "accuracy": 0.87,
                "last_updated": "2024-01-01T12:00:00Z"
            },
            {
                "model_id": "lstm_anomaly_v2.1.0",
                "model_name": "LSTM Time Series Detector",
                "model_type": "lstm_anomaly",
                "status": "deployed",
                "environment": "staging",
                "accuracy": 0.91,
                "last_updated": "2024-01-02T08:30:00Z"
            },
            {
                "model_id": "prophet_forecast_v1.2.1",
                "model_name": "Prophet Cost Forecaster",
                "model_type": "prophet_forecast",
                "status": "ready",
                "environment": "development",
                "accuracy": 0.85,
                "last_updated": "2024-01-03T14:15:00Z"
            }
        ]
        
        return {
            "models": models,
            "total_count": len(models),
            "retrieved_at": datetime.now().isoformat()
        }
    async def get_model_drift(self, model_id: str) -> dict:
        """Get model drift analysis"""
        
        drift_analysis = {
            "model_id": model_id,
            "drift_analysis": {
                "overall_drift_score": 0.34,
                "drift_threshold": 0.30,
                "drift_detected": True,
                "drift_type": "data_drift",
                "confidence_level": 0.95,
                "analysis_timestamp": datetime.now().isoformat(),
                "feature_drift": {
                    "cost_amount": {"drift_score": 0.45, "significant": True},
                    "cpu_utilization": {"drift_score": 0.28, "significant": False},
                    "memory_usage": {"drift_score": 0.52, "significant": True},
                    "network_io": {"drift_score": 0.15, "significant": False}
                },
                "recommendations": [
                    "Retrain model with recent data",
                    "Monitor cost_amount and memory_usage features closely",
                    "Consider feature engineering improvements"
                ],
                "historical_drift": [
                    {"date": "2024-01-01", "drift_score": 0.12},
                    {"date": "2024-01-02", "drift_score": 0.18},
                    {"date": "2024-01-03", "drift_score": 0.34}
                ]
            }
        }
        
        return drift_analysis
    
    async def get_audit_trail(self, event_types: Optional[str], user_id: Optional[str], 
                             start_date: str, end_date: str, limit: int) -> AuditTrailResponse:
        """Get audit trail events"""
        
        # Parse parameters
        event_type_list = event_types.split(",") if event_types else None
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Filter audit events
        filtered_events = []
        for event in self.audit_events:
            event_time_str = event["timestamp"]
            # Handle both with and without timezone info
            if event_time_str.endswith('Z'):
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
            else:
                event_time = datetime.fromisoformat(event_time_str)
                # Make timezone-aware if needed
                if event_time.tzinfo is None and start_dt.tzinfo is not None:
                    event_time = event_time.replace(tzinfo=start_dt.tzinfo)
            
            # Time range filter
            if not (start_dt <= event_time <= end_dt):
                continue
            
            # Event type filter
            if event_type_list and event["event_type"] not in event_type_list:
                continue
            
            # User ID filter
            if user_id and event.get("user_id") != user_id:
                continue
            
            filtered_events.append(event)
        
        # Apply limit
        filtered_events = filtered_events[:limit]
        
        metadata = {
            "filters": {
                "event_types": event_type_list,
                "user_id": user_id,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit
            },
            "retrieved_at": datetime.now().isoformat()
        }
        
        return AuditTrailResponse(
            events=filtered_events,
            total_count=len(filtered_events),
            metadata=metadata
        )
    
    async def get_alerts(self, account_id: str, status: Optional[str], 
                        severity: Optional[str], limit: int) -> dict:
        """Get active alerts"""
        
        # Simulate alert data
        alerts = [
            {
                "alert_id": "alert_001",
                "account_id": account_id,
                "anomaly_id": "anom_123",
                "severity": "high",
                "status": "active",
                "title": "Unusual EC2 cost spike detected",
                "description": "EC2 costs increased by 45% above baseline",
                "created_at": "2024-01-01T10:30:00Z",
                "estimated_impact_usd": 1250.75,
                "affected_services": ["EC2"],
                "notification_channels": ["email", "slack"]
            },
            {
                "alert_id": "alert_002",
                "account_id": account_id,
                "anomaly_id": "anom_124",
                "severity": "medium",
                "status": "acknowledged",
                "title": "S3 storage cost anomaly",
                "description": "S3 storage costs showing unusual pattern",
                "created_at": "2024-01-01T14:15:00Z",
                "estimated_impact_usd": 325.50,
                "affected_services": ["S3"],
                "acknowledged_by": "user@company.com",
                "acknowledged_at": "2024-01-01T15:00:00Z"
            }
        ]
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a["status"] == status]
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        alerts = alerts[:limit]
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "account_id": account_id,
            "filters": {"status": status, "severity": severity, "limit": limit},
            "retrieved_at": datetime.now().isoformat()
        }
    async def acknowledge_alert(self, alert_id: str, user_id: str, notes: Optional[str]) -> dict:
        """Acknowledge an alert"""
        
        # Log acknowledgment
        self._log_audit_event(
            event_type="alert_acknowledgment",
            description=f"Alert {alert_id} acknowledged by {user_id}",
            resource_id=alert_id,
            user_id=user_id,
            metadata={"notes": notes}
        )
        
        return {
            "alert_id": alert_id,
            "status": "acknowledged",
            "acknowledged_by": user_id,
            "acknowledged_at": datetime.now().isoformat(),
            "notes": notes,
            "message": "Alert acknowledged successfully"
        }
    
    async def snooze_alert(self, alert_id: str, duration_minutes: int, user_id: str) -> dict:
        """Snooze an alert"""
        
        snooze_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Log snooze action
        self._log_audit_event(
            event_type="alert_snooze",
            description=f"Alert {alert_id} snoozed for {duration_minutes} minutes by {user_id}",
            resource_id=alert_id,
            user_id=user_id,
            metadata={"duration_minutes": duration_minutes, "snooze_until": snooze_until.isoformat()}
        )
        
        return {
            "alert_id": alert_id,
            "status": "snoozed",
            "snoozed_by": user_id,
            "snoozed_at": datetime.now().isoformat(),
            "snooze_until": snooze_until.isoformat(),
            "duration_minutes": duration_minutes,
            "message": f"Alert snoozed for {duration_minutes} minutes"
        }
    
    async def health_check(self) -> dict:
        """API health check"""
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "uptime_seconds": 3600,  # Simulated uptime
            "dependencies": {
                "database": "healthy",
                "ml_models": "healthy",
                "notification_service": "healthy"
            }
        }
    
    async def get_system_status(self) -> dict:
        """Get system status and statistics"""
        
        return {
            "system_status": {
                "overall_status": "operational",
                "last_updated": datetime.now().isoformat(),
                "uptime_percentage": 99.9
            },
            "statistics": {
                "total_accounts_monitored": 25,
                "anomalies_detected_24h": 12,
                "alerts_generated_24h": 8,
                "forecasts_generated_24h": 45,
                "api_requests_24h": 1250,
                "average_response_time_ms": 185.7
            },
            "model_status": {
                "total_models": 3,
                "deployed_models": 2,
                "models_with_drift": 1,
                "average_accuracy": 0.88
            },
            "performance_metrics": {
                "cpu_usage_percent": 45.2,
                "memory_usage_percent": 62.8,
                "disk_usage_percent": 34.1,
                "network_throughput_mbps": 125.5
            }
        }
    
    # Helper methods
    def _simulate_anomaly_detection(self, account_id: str, start_date: datetime, end_date: datetime,
                                   services: Optional[List[str]] = None, regions: Optional[List[str]] = None,
                                   cost_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Simulate anomaly detection results"""
        
        import random
        
        anomalies = []
        num_anomalies = random.randint(3, 8)
        
        for i in range(num_anomalies):
            anomaly_time = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            anomaly = {
                "anomaly_id": f"anom_{uuid.uuid4().hex[:8]}",
                "account_id": account_id,
                "detection_timestamp": anomaly_time.isoformat(),
                "severity": random.choice(["low", "medium", "high"]),
                "confidence_score": random.uniform(0.7, 0.95),
                "anomaly_score": random.uniform(0.3, 0.9),
                "estimated_impact_usd": random.uniform(50, 2000),
                "affected_services": random.sample(
                    services or ["EC2", "S3", "RDS", "Lambda", "CloudWatch"],
                    random.randint(1, 3)
                ),
                "affected_regions": random.sample(
                    regions or ["us-east-1", "us-west-2", "eu-west-1"],
                    random.randint(1, 2)
                ),
                "description": f"Cost anomaly detected in {random.choice(['EC2', 'S3', 'RDS'])} service",
                "root_cause": random.choice([
                    "Unexpected instance scaling",
                    "Storage usage spike",
                    "Network traffic anomaly",
                    "Database query increase"
                ])
            }
            
            # Apply cost threshold filter
            if cost_threshold and anomaly["estimated_impact_usd"] < cost_threshold:
                continue
            
            anomalies.append(anomaly)
        
        return anomalies
    def _generate_forecast_data(self, account_id: str, horizon_days: int, 
                               confidence_level: float = 0.8, services: Optional[List[str]] = None,
                               granularity: str = "daily") -> List[Dict[str, Any]]:
        """Generate forecast data"""
        
        import random
        import math
        
        forecasts = []
        base_cost = random.uniform(100, 500)
        
        for i in range(horizon_days):
            forecast_date = datetime.now() + timedelta(days=i+1)
            
            # Add some trend and seasonality
            trend = i * random.uniform(0.5, 2.0)
            seasonality = 10 * math.sin(i * 0.1) if i > 0 else 0
            noise = random.uniform(-5, 5)
            
            predicted_cost = base_cost + trend + seasonality + noise
            
            forecast = {
                "date": forecast_date.date().isoformat(),
                "predicted_cost": max(0, predicted_cost),
                "confidence_level": confidence_level,
                "services": services or ["all"],
                "granularity": granularity,
                "factors": {
                    "trend": trend,
                    "seasonality": seasonality,
                    "baseline": base_cost
                }
            }
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _generate_performance_data(self, model_id: Optional[str], start_date: datetime, 
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate model performance data"""
        
        import random
        
        # Generate performance metrics
        performance_data = {
            "model_id": model_id or "isolation_forest_v1.0.0",
            "time_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "metrics": {
                "accuracy": random.uniform(0.80, 0.95),
                "precision": random.uniform(0.75, 0.90),
                "recall": random.uniform(0.80, 0.95),
                "f1_score": random.uniform(0.78, 0.92),
                "false_positive_rate": random.uniform(0.02, 0.08),
                "false_negative_rate": random.uniform(0.03, 0.10),
                "detection_latency_ms": random.uniform(50, 200)
            },
            "resource_usage": {
                "average_memory_mb": random.uniform(200, 800),
                "average_cpu_percent": random.uniform(20, 70),
                "average_prediction_time_ms": random.uniform(10, 50)
            },
            "business_impact": {
                "total_predictions": random.randint(1000, 5000),
                "anomalies_detected": random.randint(50, 200),
                "alerts_generated": random.randint(20, 80),
                "estimated_cost_savings_usd": random.uniform(5000, 25000)
            },
            "trend_analysis": {
                "accuracy_trend": "stable",
                "performance_degradation": False,
                "recommendation": "Model performing within expected parameters"
            }
        }
        
        return performance_data
    
    def _log_audit_event(self, event_type: str, description: str, resource_id: Optional[str] = None,
                        user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log audit event"""
        
        event = {
            "event_id": f"audit_{uuid.uuid4().hex[:8]}",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat() + "Z",
            "description": description,
            "resource_id": resource_id,
            "user_id": user_id or "api_system",
            "metadata": metadata or {},
            "source": "anomaly_detection_api"
        }
        
        self.audit_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.audit_events) > 1000:
            self.audit_events = self.audit_events[-1000:]

# Global API instance
anomaly_api = AnomalyDetectionAPI()

# FastAPI app instance for external use
app = anomaly_api.app if FASTAPI_AVAILABLE else None