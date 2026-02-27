"""
Multi-Cloud Cost Comparison API Models

Pydantic models for request/response validation in the multi-cloud cost comparison API.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4


class CloudProviderEnum(str, Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class ServiceCategoryEnum(str, Enum):
    """Service categories"""
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORK = "network"
    ANALYTICS = "analytics"
    SECURITY = "security"
    MANAGEMENT = "management"


class MigrationComplexityEnum(str, Enum):
    """Migration complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Request Models

class ComputeSpecRequest(BaseModel):
    """Compute specification for workload"""
    cpu_cores: int = Field(..., ge=1, le=1000, description="Number of CPU cores")
    memory_gb: int = Field(..., ge=1, le=10000, description="Memory in GB")
    instance_type_preference: Optional[str] = Field(None, description="Preferred instance type")
    operating_system: str = Field(..., description="Operating system (linux/windows)")
    architecture: str = Field(default="x86_64", description="CPU architecture")
    gpu_required: bool = Field(default=False, description="GPU requirement")
    gpu_type: Optional[str] = Field(None, description="GPU type if required")


class StorageSpecRequest(BaseModel):
    """Storage specification for workload"""
    primary_storage_gb: int = Field(..., ge=1, description="Primary storage in GB")
    storage_type: str = Field(..., description="Storage type (ssd/hdd/nvme)")
    backup_storage_gb: Optional[int] = Field(None, ge=0, description="Backup storage in GB")
    iops_requirement: Optional[int] = Field(None, ge=100, description="IOPS requirement")
    throughput_mbps: Optional[int] = Field(None, ge=1, description="Throughput in MB/s")


class NetworkSpecRequest(BaseModel):
    """Network specification for workload"""
    bandwidth_mbps: int = Field(..., ge=1, description="Required bandwidth in Mbps")
    data_transfer_gb_monthly: int = Field(..., ge=0, description="Monthly data transfer in GB")
    load_balancer_required: bool = Field(default=False, description="Load balancer requirement")
    cdn_required: bool = Field(default=False, description="CDN requirement")
    vpn_required: bool = Field(default=False, description="VPN requirement")


class DatabaseSpecRequest(BaseModel):
    """Database specification for workload"""
    database_type: str = Field(..., description="Database type (mysql/postgresql/mongodb/etc)")
    storage_gb: int = Field(..., ge=1, description="Database storage in GB")
    connections: int = Field(..., ge=1, description="Max concurrent connections")
    backup_retention_days: int = Field(default=7, ge=1, description="Backup retention in days")
    high_availability: bool = Field(default=False, description="High availability requirement")


class UsagePatternsRequest(BaseModel):
    """Usage patterns for workload"""
    hours_per_day: int = Field(..., ge=1, le=24, description="Hours of operation per day")
    days_per_week: int = Field(..., ge=1, le=7, description="Days of operation per week")
    peak_usage_multiplier: float = Field(default=1.0, ge=0.1, le=10.0, description="Peak usage multiplier")
    seasonal_variation: bool = Field(default=False, description="Seasonal usage variation")
    auto_scaling: bool = Field(default=False, description="Auto-scaling capability")


class WorkloadSpecRequest(BaseModel):
    """Complete workload specification request"""
    name: str = Field(..., min_length=1, max_length=200, description="Workload name")
    description: Optional[str] = Field(None, max_length=1000, description="Workload description")
    compute_spec: ComputeSpecRequest
    storage_spec: StorageSpecRequest
    network_spec: NetworkSpecRequest
    database_spec: Optional[DatabaseSpecRequest] = None
    additional_services: List[str] = Field(default=[], description="Additional services required")
    usage_patterns: UsagePatternsRequest
    compliance_requirements: List[str] = Field(default=[], description="Compliance requirements")
    regions: List[str] = Field(..., min_items=1, description="Preferred regions")


class MigrationRequest(BaseModel):
    """Migration analysis request"""
    workload_id: UUID4 = Field(..., description="Workload specification ID")
    source_provider: CloudProviderEnum = Field(..., description="Source cloud provider")
    target_provider: CloudProviderEnum = Field(..., description="Target cloud provider")
    migration_timeline_preference: Optional[int] = Field(None, ge=1, le=365, description="Preferred timeline in days")
    downtime_tolerance_hours: Optional[int] = Field(None, ge=0, le=168, description="Acceptable downtime in hours")
    team_size: int = Field(default=5, ge=1, le=100, description="Migration team size")
    include_training_costs: bool = Field(default=True, description="Include training costs in analysis")


class TCORequest(BaseModel):
    """TCO analysis request"""
    workload_id: UUID4 = Field(..., description="Workload specification ID")
    time_horizon_years: int = Field(default=3, ge=1, le=10, description="Analysis time horizon in years")
    include_hidden_costs: bool = Field(default=True, description="Include hidden costs")
    discount_rate: float = Field(default=0.05, ge=0.0, le=0.2, description="Discount rate for NPV calculation")


# Response Models

class CostBreakdown(BaseModel):
    """Cost breakdown by service category"""
    compute: Decimal = Field(..., description="Compute costs")
    storage: Decimal = Field(..., description="Storage costs")
    network: Decimal = Field(..., description="Network costs")
    database: Optional[Decimal] = Field(None, description="Database costs")
    additional_services: Decimal = Field(default=Decimal('0'), description="Additional service costs")
    support: Decimal = Field(default=Decimal('0'), description="Support costs")
    total: Decimal = Field(..., description="Total cost")


class CostComparisonResponse(BaseModel):
    """Cost comparison response"""
    id: UUID4 = Field(..., description="Comparison ID")
    workload_id: UUID4 = Field(..., description="Workload specification ID")
    comparison_date: datetime = Field(..., description="Comparison timestamp")
    
    # Monthly costs
    aws_monthly_cost: Optional[Decimal] = Field(None, description="AWS monthly cost")
    gcp_monthly_cost: Optional[Decimal] = Field(None, description="GCP monthly cost")
    azure_monthly_cost: Optional[Decimal] = Field(None, description="Azure monthly cost")
    
    # Annual costs
    aws_annual_cost: Optional[Decimal] = Field(None, description="AWS annual cost")
    gcp_annual_cost: Optional[Decimal] = Field(None, description="GCP annual cost")
    azure_annual_cost: Optional[Decimal] = Field(None, description="Azure annual cost")
    
    # Detailed breakdown
    cost_breakdown: Dict[str, CostBreakdown] = Field(..., description="Cost breakdown by provider")
    recommendations: List[str] = Field(default=[], description="Cost optimization recommendations")
    pricing_data_version: Optional[str] = Field(None, description="Pricing data version used")
    
    # Comparison insights
    lowest_cost_provider: Optional[str] = Field(None, description="Provider with lowest cost")
    cost_difference_percentage: Optional[Dict[str, float]] = Field(None, description="Cost differences as percentages")


class TCOAnalysisResponse(BaseModel):
    """TCO analysis response"""
    id: UUID4 = Field(..., description="Analysis ID")
    workload_id: UUID4 = Field(..., description="Workload specification ID")
    analysis_date: datetime = Field(..., description="Analysis timestamp")
    time_horizon_years: int = Field(..., description="Analysis time horizon")
    
    # TCO by provider
    aws_tco: Dict[str, Any] = Field(..., description="AWS TCO breakdown")
    gcp_tco: Dict[str, Any] = Field(..., description="GCP TCO breakdown")
    azure_tco: Dict[str, Any] = Field(..., description="Azure TCO breakdown")
    
    # Cost components
    hidden_costs: Dict[str, Any] = Field(..., description="Hidden costs analysis")
    operational_costs: Dict[str, Any] = Field(..., description="Operational costs")
    cost_projections: Dict[str, Any] = Field(..., description="Year-by-year projections")
    
    # Summary
    total_tco_comparison: Dict[str, Decimal] = Field(..., description="Total TCO by provider")
    recommended_provider: Optional[str] = Field(None, description="Recommended provider based on TCO")


class RiskAssessment(BaseModel):
    """Migration risk assessment"""
    overall_risk_level: str = Field(..., description="Overall risk level")
    technical_risks: List[str] = Field(default=[], description="Technical risks")
    business_risks: List[str] = Field(default=[], description="Business risks")
    mitigation_strategies: List[str] = Field(default=[], description="Risk mitigation strategies")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Migration success probability")


class MigrationAnalysisResponse(BaseModel):
    """Migration analysis response"""
    id: UUID4 = Field(..., description="Analysis ID")
    workload_id: UUID4 = Field(..., description="Workload specification ID")
    source_provider: str = Field(..., description="Source provider")
    target_provider: str = Field(..., description="Target provider")
    analysis_date: datetime = Field(..., description="Analysis timestamp")
    
    # Cost analysis
    migration_cost: Decimal = Field(..., description="Total migration cost")
    migration_timeline_days: int = Field(..., description="Estimated timeline in days")
    break_even_months: Optional[int] = Field(None, description="Break-even period in months")
    
    # Detailed breakdown
    cost_breakdown: Dict[str, Decimal] = Field(..., description="Migration cost breakdown")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    recommendations: List[str] = Field(..., description="Migration recommendations")
    
    # ROI analysis
    monthly_savings: Optional[Decimal] = Field(None, description="Expected monthly savings")
    annual_savings: Optional[Decimal] = Field(None, description="Expected annual savings")
    roi_percentage: Optional[float] = Field(None, description="Return on investment percentage")


class ServicePricing(BaseModel):
    """Service pricing information"""
    provider: str = Field(..., description="Cloud provider")
    service_name: str = Field(..., description="Service name")
    service_category: str = Field(..., description="Service category")
    region: str = Field(..., description="Region")
    pricing_unit: str = Field(..., description="Pricing unit")
    price_per_unit: Decimal = Field(..., description="Price per unit")
    currency: str = Field(..., description="Currency")
    effective_date: datetime = Field(..., description="Effective date")
    pricing_details: Optional[Dict[str, Any]] = Field(None, description="Additional pricing details")


class CloudProvider(BaseModel):
    """Cloud provider information"""
    name: str = Field(..., description="Provider name")
    provider_type: str = Field(..., description="Provider type")
    supported_regions: List[str] = Field(..., description="Supported regions")
    supported_services: List[str] = Field(..., description="Supported services")
    pricing_model: str = Field(..., description="Pricing model")


class CloudService(BaseModel):
    """Cloud service information"""
    name: str = Field(..., description="Service name")
    category: str = Field(..., description="Service category")
    description: str = Field(..., description="Service description")
    pricing_units: List[str] = Field(..., description="Available pricing units")
    regions: List[str] = Field(..., description="Available regions")


class ServiceEquivalencyResponse(BaseModel):
    """Service equivalency information"""
    source_provider: str = Field(..., description="Source provider")
    source_service: str = Field(..., description="Source service")
    target_provider: str = Field(..., description="Target provider")
    target_service: str = Field(..., description="Target service")
    service_category: str = Field(..., description="Service category")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    feature_parity_score: float = Field(..., ge=0.0, le=1.0, description="Feature parity score")
    performance_ratio: float = Field(..., description="Performance ratio")
    cost_efficiency_score: float = Field(..., ge=0.0, le=1.0, description="Cost efficiency score")
    migration_complexity: str = Field(..., description="Migration complexity")
    limitations: List[str] = Field(default=[], description="Service limitations")
    additional_features: List[str] = Field(default=[], description="Additional features")
    mapping_notes: Optional[str] = Field(None, description="Mapping notes")


# Error Models

class ErrorDetail(BaseModel):
    """Error detail information"""
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Error details")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(..., description="Error timestamp")


# Validation Models

class WorkloadValidationResponse(BaseModel):
    """Workload validation response"""
    is_valid: bool = Field(..., description="Validation result")
    errors: List[ErrorDetail] = Field(default=[], description="Validation errors")
    warnings: List[str] = Field(default=[], description="Validation warnings")
    estimated_monthly_cost_range: Optional[Dict[str, Decimal]] = Field(None, description="Estimated cost range")


# List Response Models

class WorkloadListResponse(BaseModel):
    """Workload list response"""
    workloads: List[Dict[str, Any]] = Field(..., description="List of workloads")
    total_count: int = Field(..., description="Total number of workloads")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


class ComparisonListResponse(BaseModel):
    """Cost comparison list response"""
    comparisons: List[CostComparisonResponse] = Field(..., description="List of cost comparisons")
    total_count: int = Field(..., description="Total number of comparisons")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


# Utility validators

@validator('regions', each_item=True)
def validate_region(cls, v):
    """Validate region format"""
    if not v or len(v) < 2:
        raise ValueError('Region must be at least 2 characters long')
    return v.lower()


@validator('additional_services', each_item=True)
def validate_service_name(cls, v):
    """Validate service name format"""
    if not v or len(v.strip()) == 0:
        raise ValueError('Service name cannot be empty')
    return v.strip()