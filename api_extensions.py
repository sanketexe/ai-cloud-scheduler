# api_extensions.py
"""
API Extensions for Cloud Intelligence Platform

This module extends the existing REST API with:
- FinOps cost management endpoints
- Performance monitoring endpoints  
- Simulation endpoints for running scenarios

Requirements addressed:
- 5.1: REST API for all major functions
- 5.2: API authentication and authorization
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query, Path, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from dataclasses import asdict

# Import existing modules
from finops_engine import (
    FinOpsEngine, CostData, Budget, BudgetStatus, CostOptimization, 
    CostForecast, SpendingAnalysis, CloudProvider as FinOpsCloudProvider,
    CostCategory, BudgetPeriod, OptimizationType
)
from performance_monitoring_system import (
    PerformanceMonitoringSystem, MonitoringConfig, MonitoringReport,
    CloudResource, MetricType, ResourceCapacity
)
from simulation_framework import (
    SimulationEngine, WorkloadPattern, SimulationConfig, SimulationResults,
    PatternType, ScenarioConfig
)
from enhanced_models import (
    EnhancedWorkload, CostConstraints, PerformanceRequirements, 
    ComplianceRequirements, WorkloadPriority, CostOptimizationLevel
)

# Security
security = HTTPBearer()

# Pydantic Models for API

class CloudProviderEnum(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    OTHER = "other"

class CostCategoryEnum(str, Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"
    OTHER = "other"

class BudgetPeriodEnum(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

# Cost Management Models
class CostDataRequest(BaseModel):
    provider: CloudProviderEnum
    start_date: datetime
    end_date: datetime
    filters: Optional[Dict[str, Any]] = None

class CostDataResponse(BaseModel):
    provider: str
    service: str
    resource_id: str
    cost_amount: float
    currency: str
    billing_period_start: datetime
    billing_period_end: datetime
    cost_category: str
    tags: Dict[str, str] = {}
    region: Optional[str] = None

class BudgetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0)
    currency: str = Field(default="USD", regex="^[A-Z]{3}$")
    period: BudgetPeriodEnum
    start_date: datetime
    end_date: Optional[datetime] = None
    alert_thresholds: List[float] = Field(default=[50.0, 80.0, 100.0])
    scope_filters: Dict[str, List[str]] = {}

    @validator('alert_thresholds')
    def validate_thresholds(cls, v):
        if not all(0 <= threshold <= 100 for threshold in v):
            raise ValueError('Alert thresholds must be between 0 and 100')
        return sorted(v)

class BudgetResponse(BaseModel):
    budget_id: str
    name: str
    amount: float
    currency: str
    period: str
    start_date: datetime
    end_date: Optional[datetime]
    alert_thresholds: List[float]
    scope_filters: Dict[str, List[str]]
    is_active: bool
    created_at: datetime

class BudgetStatusResponse(BaseModel):
    budget_id: str
    budget_name: str
    current_spend: float
    utilization_percentage: float
    projected_spend: float
    days_remaining: int
    is_on_track: bool
    triggered_alerts: List[str]
    last_updated: datetime

class CostOptimizationResponse(BaseModel):
    optimization_id: str
    optimization_type: str
    resource_id: str
    current_cost: float
    optimized_cost: float
    potential_savings: float
    confidence_score: float
    implementation_effort: str
    recommendation: str
    created_at: datetime

class CostForecastRequest(BaseModel):
    forecast_days: int = Field(default=30, ge=1, le=365)
    providers: Optional[List[CloudProviderEnum]] = None
    include_confidence_intervals: bool = True

class CostForecastResponse(BaseModel):
    forecast_period_start: datetime
    forecast_period_end: datetime
    predicted_costs: List[float]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    model_accuracy: float
    trend_direction: str

# Performance Monitoring Models
class MetricTypeEnum(str, Enum):
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"

class CloudResourceRequest(BaseModel):
    resource_id: str
    resource_type: str
    provider: CloudProviderEnum
    region: str
    tags: Dict[str, str] = {}

class ResourceCapacityRequest(BaseModel):
    cpu_cores: int = Field(..., ge=1)
    memory_gb: float = Field(..., gt=0)
    disk_gb: float = Field(..., gt=0)
    network_bandwidth_mbps: float = Field(..., gt=0)
    instance_type: str

class MetricsRequest(BaseModel):
    resource_ids: List[str]
    metric_types: List[MetricTypeEnum]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class MetricsResponse(BaseModel):
    resource_id: str
    metric_type: str
    value: float
    timestamp: datetime
    unit: str

class MonitoringConfigRequest(BaseModel):
    metrics_collection_interval_minutes: int = Field(default=5, ge=1, le=60)
    anomaly_detection_enabled: bool = True
    anomaly_sensitivity: float = Field(default=0.1, ge=0.01, le=1.0)
    health_check_interval_minutes: int = Field(default=5, ge=1, le=60)
    trend_analysis_period_days: int = Field(default=30, ge=1, le=365)
    capacity_forecast_days: int = Field(default=90, ge=1, le=365)
    alert_cooldown_minutes: int = Field(default=15, ge=1, le=1440)

class MonitoringReportResponse(BaseModel):
    report_id: str
    generated_at: datetime
    time_period_start: datetime
    time_period_end: datetime
    total_resources_monitored: int
    total_metrics_collected: int
    healthy_resources: int
    warning_resources: int
    unhealthy_resources: int
    total_anomalies_detected: int
    total_alerts_triggered: int
    resources_needing_scaling: int
    optimization_opportunities: List[str]

# Simulation Models
class PatternTypeEnum(str, Enum):
    CONSTANT = "constant"
    PERIODIC = "periodic"
    BURSTY = "bursty"
    RANDOM_WALK = "random_walk"

class WorkloadPatternRequest(BaseModel):
    pattern_type: PatternTypeEnum
    base_intensity: float = Field(..., gt=0)
    variation_amplitude: float = Field(default=0.1, ge=0, le=1)
    noise_level: float = Field(default=0.05, ge=0, le=1)

class SimulationRequest(BaseModel):
    scenario_name: str
    workload_pattern: WorkloadPatternRequest
    duration_hours: int = Field(..., ge=1, le=8760)  # Max 1 year
    scheduler_type: str
    environment_constraints: Dict[str, Any] = {}

class SimulationResultsResponse(BaseModel):
    simulation_id: str
    scenario_name: str
    start_time: datetime
    end_time: datetime
    total_workloads: int
    successful_placements: int
    failed_placements: int
    total_cost: float
    average_utilization: float
    performance_score: float

# API Extension Class
class APIExtensions:
    """Extended API endpoints for FinOps and Performance Monitoring"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.logger = logging.getLogger(f"{__name__}.APIExtensions")
        
        # Initialize engines
        self.finops_engine = FinOpsEngine()
        self.monitoring_system = PerformanceMonitoringSystem()
        self.simulation_engine = SimulationEngine()
        
        # Add routes
        self._add_cost_management_routes()
        self._add_performance_monitoring_routes()
        self._add_simulation_routes()
        
        self.logger.info("API Extensions initialized")
    
    def _add_cost_management_routes(self):
        """Add cost management endpoints"""
        
        @self.app.post("/api/v1/costs/collect", response_model=List[CostDataResponse])
        async def collect_cost_data(
            request: CostDataRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Collect cost data from cloud providers"""
            try:
                # Validate token (simplified)
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Convert enum to internal format
                provider = FinOpsCloudProvider(request.provider.value)
                
                # Collect cost data
                cost_data = await self.finops_engine.collect_cost_data(
                    providers=[provider],
                    start_date=request.start_date,
                    end_date=request.end_date,
                    filters=request.filters
                )
                
                # Convert to response format
                response_data = []
                for data in cost_data:
                    response_data.append(CostDataResponse(
                        provider=data.provider.value,
                        service=data.service,
                        resource_id=data.resource_id,
                        cost_amount=data.cost_amount,
                        currency=data.currency,
                        billing_period_start=data.billing_period_start,
                        billing_period_end=data.billing_period_end,
                        cost_category=data.cost_category.value,
                        tags=data.tags,
                        region=data.region
                    ))
                
                return response_data
                
            except Exception as e:
                self.logger.error(f"Error collecting cost data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/budgets", response_model=BudgetResponse)
        async def create_budget(
            request: BudgetCreateRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Create a new budget"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                budget = await self.finops_engine.create_budget(
                    name=request.name,
                    amount=request.amount,
                    currency=request.currency,
                    period=BudgetPeriod(request.period.value),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    alert_thresholds=request.alert_thresholds,
                    scope_filters=request.scope_filters
                )
                
                return BudgetResponse(
                    budget_id=budget.budget_id,
                    name=budget.name,
                    amount=budget.amount,
                    currency=budget.currency,
                    period=budget.period.value,
                    start_date=budget.start_date,
                    end_date=budget.end_date,
                    alert_thresholds=budget.alert_thresholds,
                    scope_filters=budget.scope_filters,
                    is_active=budget.is_active,
                    created_at=budget.created_at
                )
                
            except Exception as e:
                self.logger.error(f"Error creating budget: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/budgets", response_model=List[BudgetResponse])
        async def list_budgets(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            active_only: bool = Query(True, description="Return only active budgets")
        ):
            """List all budgets"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                budgets = await self.finops_engine.list_budgets(active_only=active_only)
                
                return [
                    BudgetResponse(
                        budget_id=budget.budget_id,
                        name=budget.name,
                        amount=budget.amount,
                        currency=budget.currency,
                        period=budget.period.value,
                        start_date=budget.start_date,
                        end_date=budget.end_date,
                        alert_thresholds=budget.alert_thresholds,
                        scope_filters=budget.scope_filters,
                        is_active=budget.is_active,
                        created_at=budget.created_at
                    )
                    for budget in budgets
                ]
                
            except Exception as e:
                self.logger.error(f"Error listing budgets: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/budgets/{budget_id}/status", response_model=BudgetStatusResponse)
        async def get_budget_status(
            budget_id: str = Path(..., description="Budget ID"),
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get budget utilization status"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                status = await self.finops_engine.get_budget_status(budget_id)
                
                return BudgetStatusResponse(
                    budget_id=status.budget.budget_id,
                    budget_name=status.budget.name,
                    current_spend=status.current_spend,
                    utilization_percentage=status.utilization_percentage,
                    projected_spend=status.projected_spend,
                    days_remaining=status.days_remaining,
                    is_on_track=status.is_on_track,
                    triggered_alerts=status.triggered_alerts,
                    last_updated=status.last_updated
                )
                
            except Exception as e:
                self.logger.error(f"Error getting budget status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/costs/optimizations", response_model=List[CostOptimizationResponse])
        async def get_cost_optimizations(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            min_savings: float = Query(0, ge=0, description="Minimum potential savings"),
            optimization_type: Optional[str] = Query(None, description="Filter by optimization type")
        ):
            """Get cost optimization recommendations"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                optimizations = await self.finops_engine.get_optimization_recommendations(
                    min_savings=min_savings,
                    optimization_type=optimization_type
                )
                
                return [
                    CostOptimizationResponse(
                        optimization_id=opt.optimization_id,
                        optimization_type=opt.optimization_type.value,
                        resource_id=opt.resource_id,
                        current_cost=opt.current_cost,
                        optimized_cost=opt.optimized_cost,
                        potential_savings=opt.potential_savings,
                        confidence_score=opt.confidence_score,
                        implementation_effort=opt.implementation_effort.value,
                        recommendation=opt.recommendation,
                        created_at=opt.created_at
                    )
                    for opt in optimizations
                ]
                
            except Exception as e:
                self.logger.error(f"Error getting cost optimizations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/costs/forecast", response_model=CostForecastResponse)
        async def generate_cost_forecast(
            request: CostForecastRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Generate cost forecast"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                providers = None
                if request.providers:
                    providers = [FinOpsCloudProvider(p.value) for p in request.providers]
                
                forecast = await self.finops_engine.generate_cost_forecast(
                    forecast_days=request.forecast_days,
                    providers=providers,
                    include_confidence_intervals=request.include_confidence_intervals
                )
                
                confidence_intervals = None
                if request.include_confidence_intervals and forecast.confidence_intervals:
                    confidence_intervals = [
                        {"lower": lower, "upper": upper}
                        for lower, upper in forecast.confidence_intervals
                    ]
                
                return CostForecastResponse(
                    forecast_period_start=forecast.forecast_period_start,
                    forecast_period_end=forecast.forecast_period_end,
                    predicted_costs=forecast.predicted_costs,
                    confidence_intervals=confidence_intervals,
                    model_accuracy=forecast.model_accuracy,
                    trend_direction=forecast.trend_direction
                )
                
            except Exception as e:
                self.logger.error(f"Error generating cost forecast: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_performance_monitoring_routes(self):
        """Add performance monitoring endpoints"""
        
        @self.app.post("/api/v1/monitoring/resources")
        async def add_monitoring_resources(
            resources: List[CloudResourceRequest],
            capacities: Optional[Dict[str, ResourceCapacityRequest]] = None,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Add resources to monitoring"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Convert to internal format
                cloud_resources = []
                for resource in resources:
                    cloud_resources.append(CloudResource(
                        resource_id=resource.resource_id,
                        resource_type=resource.resource_type,
                        provider=resource.provider.value,
                        region=resource.region,
                        tags=resource.tags
                    ))
                
                resource_capacities = {}
                if capacities:
                    for resource_id, capacity in capacities.items():
                        resource_capacities[resource_id] = ResourceCapacity(
                            cpu_cores=capacity.cpu_cores,
                            memory_gb=capacity.memory_gb,
                            disk_gb=capacity.disk_gb,
                            network_bandwidth_mbps=capacity.network_bandwidth_mbps,
                            instance_type=capacity.instance_type
                        )
                
                self.monitoring_system.add_resources_to_monitor(
                    cloud_resources, resource_capacities
                )
                
                return {
                    "message": f"Added {len(resources)} resources to monitoring",
                    "resource_ids": [r.resource_id for r in resources]
                }
                
            except Exception as e:
                self.logger.error(f"Error adding monitoring resources: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/monitoring/metrics/collect", response_model=List[MetricsResponse])
        async def collect_metrics(
            request: MetricsRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Collect performance metrics"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Get resources
                resources = [
                    self.monitoring_system.monitored_resources[rid]
                    for rid in request.resource_ids
                    if rid in self.monitoring_system.monitored_resources
                ]
                
                if not resources:
                    raise HTTPException(status_code=404, detail="No valid resources found")
                
                # Convert metric types
                metric_types = [MetricType(mt.value) for mt in request.metric_types]
                
                # Collect metrics
                metrics_data = await self.monitoring_system.metrics_collector.collect_metrics(
                    resources=resources,
                    metric_types=metric_types,
                    start_time=request.start_time,
                    end_time=request.end_time
                )
                
                # Convert to response format
                response_data = []
                for resource_id, data in metrics_data.items():
                    for metric in data.metrics:
                        response_data.append(MetricsResponse(
                            resource_id=resource_id,
                            metric_type=metric.metric_type.value,
                            value=metric.value,
                            timestamp=metric.timestamp,
                            unit=metric.unit
                        ))
                
                return response_data
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/monitoring/status")
        async def get_monitoring_status(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get monitoring system status"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                return self.monitoring_system.get_system_status()
                
            except Exception as e:
                self.logger.error(f"Error getting monitoring status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/monitoring/start")
        async def start_monitoring(
            config: Optional[MonitoringConfigRequest] = None,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Start monitoring system"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                if config:
                    monitoring_config = MonitoringConfig(
                        metrics_collection_interval_minutes=config.metrics_collection_interval_minutes,
                        anomaly_detection_enabled=config.anomaly_detection_enabled,
                        anomaly_sensitivity=config.anomaly_sensitivity,
                        health_check_interval_minutes=config.health_check_interval_minutes,
                        trend_analysis_period_days=config.trend_analysis_period_days,
                        capacity_forecast_days=config.capacity_forecast_days,
                        alert_cooldown_minutes=config.alert_cooldown_minutes
                    )
                    self.monitoring_system.config = monitoring_config
                
                await self.monitoring_system.start_monitoring()
                
                return {"message": "Monitoring system started", "status": "running"}
                
            except Exception as e:
                self.logger.error(f"Error starting monitoring: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/monitoring/stop")
        async def stop_monitoring(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Stop monitoring system"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                await self.monitoring_system.stop_monitoring()
                
                return {"message": "Monitoring system stopped", "status": "stopped"}
                
            except Exception as e:
                self.logger.error(f"Error stopping monitoring: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/monitoring/reports/{report_id}", response_model=MonitoringReportResponse)
        async def get_monitoring_report(
            report_id: str = Path(..., description="Report ID"),
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get monitoring report by ID"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # In a real implementation, this would retrieve from storage
                # For now, generate a new report
                report = await self.monitoring_system.generate_monitoring_report()
                
                return MonitoringReportResponse(
                    report_id=report.report_id,
                    generated_at=report.generated_at,
                    time_period_start=report.time_period_start,
                    time_period_end=report.time_period_end,
                    total_resources_monitored=report.total_resources_monitored,
                    total_metrics_collected=report.total_metrics_collected,
                    healthy_resources=report.healthy_resources,
                    warning_resources=report.warning_resources,
                    unhealthy_resources=report.unhealthy_resources,
                    total_anomalies_detected=report.total_anomalies_detected,
                    total_alerts_triggered=report.total_alerts_triggered,
                    resources_needing_scaling=report.resources_needing_scaling,
                    optimization_opportunities=report.optimization_opportunities
                )
                
            except Exception as e:
                self.logger.error(f"Error getting monitoring report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/monitoring/reports/generate", response_model=MonitoringReportResponse)
        async def generate_monitoring_report(
            time_period_hours: int = Query(24, ge=1, le=8760, description="Report time period in hours"),
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Generate new monitoring report"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                report = await self.monitoring_system.generate_monitoring_report(time_period_hours)
                
                return MonitoringReportResponse(
                    report_id=report.report_id,
                    generated_at=report.generated_at,
                    time_period_start=report.time_period_start,
                    time_period_end=report.time_period_end,
                    total_resources_monitored=report.total_resources_monitored,
                    total_metrics_collected=report.total_metrics_collected,
                    healthy_resources=report.healthy_resources,
                    warning_resources=report.warning_resources,
                    unhealthy_resources=report.unhealthy_resources,
                    total_anomalies_detected=report.total_anomalies_detected,
                    total_alerts_triggered=report.total_alerts_triggered,
                    resources_needing_scaling=report.resources_needing_scaling,
                    optimization_opportunities=report.optimization_opportunities
                )
                
            except Exception as e:
                self.logger.error(f"Error generating monitoring report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_simulation_routes(self):
        """Add simulation endpoints"""
        
        @self.app.post("/api/v1/simulations/run", response_model=SimulationResultsResponse)
        async def run_simulation(
            request: SimulationRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Run simulation scenario"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Convert workload pattern
                workload_pattern = WorkloadPattern(
                    pattern_type=PatternType(request.workload_pattern.pattern_type.value),
                    base_intensity=request.workload_pattern.base_intensity,
                    variation_amplitude=request.workload_pattern.variation_amplitude,
                    noise_level=request.workload_pattern.noise_level
                )
                
                # Create simulation config
                config = SimulationConfig(
                    scenario_name=request.scenario_name,
                    workload_pattern=workload_pattern,
                    duration_hours=request.duration_hours,
                    scheduler_type=request.scheduler_type,
                    environment_constraints=request.environment_constraints
                )
                
                # Run simulation
                results = await self.simulation_engine.run_simulation(config)
                
                return SimulationResultsResponse(
                    simulation_id=results.simulation_id,
                    scenario_name=results.scenario_name,
                    start_time=results.start_time,
                    end_time=results.end_time,
                    total_workloads=results.total_workloads,
                    successful_placements=results.successful_placements,
                    failed_placements=results.failed_placements,
                    total_cost=results.total_cost,
                    average_utilization=results.average_utilization,
                    performance_score=results.performance_score
                )
                
            except Exception as e:
                self.logger.error(f"Error running simulation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/simulations/{simulation_id}/results", response_model=SimulationResultsResponse)
        async def get_simulation_results(
            simulation_id: str = Path(..., description="Simulation ID"),
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get simulation results by ID"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                results = await self.simulation_engine.get_simulation_results(simulation_id)
                
                if not results:
                    raise HTTPException(status_code=404, detail="Simulation not found")
                
                return SimulationResultsResponse(
                    simulation_id=results.simulation_id,
                    scenario_name=results.scenario_name,
                    start_time=results.start_time,
                    end_time=results.end_time,
                    total_workloads=results.total_workloads,
                    successful_placements=results.successful_placements,
                    failed_placements=results.failed_placements,
                    total_cost=results.total_cost,
                    average_utilization=results.average_utilization,
                    performance_score=results.performance_score
                )
                
            except Exception as e:
                self.logger.error(f"Error getting simulation results: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/simulations")
        async def list_simulations(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
            offset: int = Query(0, ge=0, description="Number of results to skip")
        ):
            """List simulation runs"""
            try:
                if not self._validate_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                simulations = await self.simulation_engine.list_simulations(
                    limit=limit, offset=offset
                )
                
                return {
                    "simulations": simulations,
                    "total": len(simulations),
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                self.logger.error(f"Error listing simulations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token (simplified implementation)"""
        # In a real implementation, this would validate JWT tokens, API keys, etc.
        # For demonstration, accept any non-empty token
        return bool(token and len(token) > 0)


# Function to extend existing FastAPI app
def extend_api(app: FastAPI) -> APIExtensions:
    """Extend existing FastAPI app with new endpoints"""
    
    # Add CORS middleware if not already added
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize and return extensions
    extensions = APIExtensions(app)
    
    # Add health check for new endpoints
    @app.get("/api/v1/health")
    async def health_check_v1():
        """Health check for API v1 endpoints"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "finops_engine": "available",
                "monitoring_system": "available", 
                "simulation_engine": "available"
            }
        }
    
    return extensions