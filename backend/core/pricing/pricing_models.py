"""
Pricing data models for multi-cloud cost comparison.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PricingUnit(str, Enum):
    """Pricing unit types"""
    HOUR = "hour"
    MONTH = "month"
    GB_MONTH = "gb-month"
    GB = "gb"
    REQUEST = "request"
    VCPU_HOUR = "vcpu-hour"
    INSTANCE_HOUR = "instance-hour"


class ServiceCategory(str, Enum):
    """Service category types"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CONTAINER = "container"
    SERVERLESS = "serverless"


@dataclass
class ComputePricing:
    """Compute service pricing information"""
    instance_type: str
    vcpus: int
    memory_gb: float
    price_per_hour: Decimal
    price_per_month: Decimal
    operating_system: str
    region: str
    currency: str = "USD"
    spot_price_per_hour: Optional[Decimal] = None
    reserved_price_per_hour: Optional[Decimal] = None
    architecture: str = "x86_64"
    network_performance: Optional[str] = None
    storage_type: Optional[str] = None
    additional_specs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_specs is None:
            self.additional_specs = {}


@dataclass
class StoragePricing:
    """Storage service pricing information"""
    storage_type: str  # object, block, file, etc.
    price_per_gb_month: Decimal
    region: str
    currency: str = "USD"
    storage_class: Optional[str] = None  # standard, infrequent, archive
    iops_price: Optional[Decimal] = None
    throughput_price: Optional[Decimal] = None
    request_price: Optional[Decimal] = None
    retrieval_price: Optional[Decimal] = None
    minimum_storage_duration: Optional[int] = None  # days
    additional_specs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_specs is None:
            self.additional_specs = {}


@dataclass
class NetworkPricing:
    """Network service pricing information"""
    service_type: str  # data_transfer, load_balancer, cdn, etc.
    price_per_gb: Optional[Decimal] = None
    price_per_hour: Optional[Decimal] = None
    price_per_request: Optional[Decimal] = None
    region: str = ""
    currency: str = "USD"
    transfer_type: Optional[str] = None  # inbound, outbound, inter_region
    bandwidth_tier: Optional[str] = None
    additional_specs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_specs is None:
            self.additional_specs = {}


@dataclass
class DatabasePricing:
    """Database service pricing information"""
    database_type: str  # mysql, postgresql, mongodb, etc.
    instance_class: str
    price_per_hour: Decimal
    storage_price_per_gb_month: Decimal
    region: str
    currency: str = "USD"
    engine_version: Optional[str] = None
    multi_az: bool = False
    backup_storage_price: Optional[Decimal] = None
    iops_price: Optional[Decimal] = None
    additional_specs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_specs is None:
            self.additional_specs = {}


@dataclass
class PricingData:
    """Comprehensive pricing data for a service"""
    provider: str
    service_name: str
    service_category: ServiceCategory
    region: str
    pricing_date: datetime
    compute_pricing: Optional[List[ComputePricing]] = None
    storage_pricing: Optional[List[StoragePricing]] = None
    network_pricing: Optional[List[NetworkPricing]] = None
    database_pricing: Optional[List[DatabasePricing]] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.compute_pricing is None:
            self.compute_pricing = []
        if self.storage_pricing is None:
            self.storage_pricing = []
        if self.network_pricing is None:
            self.network_pricing = []
        if self.database_pricing is None:
            self.database_pricing = []
        if self.raw_data is None:
            self.raw_data = {}


@dataclass
class PricingQuery:
    """Pricing query parameters"""
    provider: str
    service_name: str
    region: str
    instance_type: Optional[str] = None
    storage_type: Optional[str] = None
    operating_system: Optional[str] = None
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class PricingResponse:
    """Pricing API response"""
    success: bool
    data: Optional[PricingData] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None
    cached: bool = False
    cache_age_seconds: Optional[int] = None