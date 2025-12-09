"""
Cloud Provider Catalog Data Structures

This module defines the data structures for cloud provider catalogs including
service offerings, pricing models, compliance certifications, and capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from decimal import Decimal


class CloudProviderName(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class ServiceCategory(Enum):
    """Cloud service categories"""
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORKING = "networking"
    MACHINE_LEARNING = "machine_learning"
    ANALYTICS = "analytics"
    CONTAINERS = "containers"
    SERVERLESS = "serverless"
    SECURITY = "security"
    MANAGEMENT = "management"
    IOT = "iot"
    BLOCKCHAIN = "blockchain"
    QUANTUM = "quantum"
    MEDIA = "media"
    DEVELOPER_TOOLS = "developer_tools"


class PricingModel(Enum):
    """Pricing model types"""
    ON_DEMAND = "on_demand"
    RESERVED = "reserved"
    SPOT = "spot"
    SAVINGS_PLAN = "savings_plan"
    COMMITTED_USE = "committed_use"
    PREEMPTIBLE = "preemptible"


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    FedRAMP = "fedramp"
    HITRUST = "hitrust"
    FISMA = "fisma"
    CCPA = "ccpa"
    NIST = "nist"


@dataclass
class ServiceSpecification:
    """Detailed service specification"""
    service_id: str
    service_name: str
    category: ServiceCategory
    description: str
    features: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    regions_available: List[str] = field(default_factory=list)
    sla_percentage: Optional[float] = None
    integration_capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"<ServiceSpecification(id='{self.service_id}', name='{self.service_name}')>"


@dataclass
class PricingTier:
    """Pricing tier information"""
    tier_name: str
    pricing_model: PricingModel
    unit: str  # e.g., "per hour", "per GB", "per request"
    price_per_unit: Decimal
    minimum_commitment: Optional[str] = None
    discount_percentage: Optional[float] = None
    conditions: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"<PricingTier(name='{self.tier_name}', price={self.price_per_unit}/{self.unit})>"


@dataclass
class ServicePricing:
    """Pricing information for a service"""
    service_id: str
    region: str
    pricing_tiers: List[PricingTier] = field(default_factory=list)
    free_tier: Optional[Dict[str, Any]] = None
    additional_costs: Dict[str, Decimal] = field(default_factory=dict)  # data transfer, storage, etc.
    currency: str = "USD"
    last_updated: Optional[str] = None
    
    def __repr__(self):
        return f"<ServicePricing(service='{self.service_id}', region='{self.region}')>"


@dataclass
class ComplianceCertification:
    """Compliance certification details"""
    framework: ComplianceFramework
    certification_name: str
    description: str
    regions_covered: List[str] = field(default_factory=list)
    services_covered: List[str] = field(default_factory=list)
    certification_date: Optional[str] = None
    expiry_date: Optional[str] = None
    audit_report_available: bool = False
    documentation_url: Optional[str] = None
    
    def __repr__(self):
        return f"<ComplianceCertification(framework='{self.framework.value}', name='{self.certification_name}')>"


@dataclass
class RegionSpecification:
    """Cloud region specification"""
    region_id: str
    region_name: str
    geographic_location: str
    availability_zones: int
    services_available: List[str] = field(default_factory=list)
    compliance_certifications: List[str] = field(default_factory=list)
    data_residency_compliant: bool = True
    latency_zones: Dict[str, int] = field(default_factory=dict)  # region -> avg latency ms
    
    def __repr__(self):
        return f"<RegionSpecification(id='{self.region_id}', name='{self.region_name}')>"


@dataclass
class PerformanceCapability:
    """Performance capabilities of a provider"""
    max_compute_instances: Optional[int] = None
    max_storage_capacity_tb: Optional[int] = None
    max_network_bandwidth_gbps: Optional[int] = None
    gpu_availability: bool = False
    gpu_types: List[str] = field(default_factory=list)
    specialized_compute: List[str] = field(default_factory=list)  # HPC, quantum, etc.
    auto_scaling_capabilities: bool = True
    load_balancing_options: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"<PerformanceCapability(gpu={self.gpu_availability}, auto_scale={self.auto_scaling_capabilities})>"


@dataclass
class SupportTier:
    """Support tier information"""
    tier_name: str
    response_time_critical: str  # e.g., "15 minutes", "1 hour"
    response_time_high: str
    response_time_normal: str
    channels: List[str] = field(default_factory=list)  # phone, email, chat, etc.
    technical_account_manager: bool = False
    monthly_cost: Optional[Decimal] = None
    features: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"<SupportTier(name='{self.tier_name}', critical_response='{self.response_time_critical}')>"


@dataclass
class CloudProvider:
    """Complete cloud provider catalog"""
    provider_name: CloudProviderName
    display_name: str
    description: str
    headquarters: str
    founded_year: int
    
    # Service catalog
    services: Dict[str, ServiceSpecification] = field(default_factory=dict)
    service_categories: Dict[ServiceCategory, List[str]] = field(default_factory=dict)
    
    # Pricing information
    pricing_data: Dict[str, ServicePricing] = field(default_factory=dict)
    
    # Compliance and certifications
    compliance_certifications: List[ComplianceCertification] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Regional presence
    regions: List[RegionSpecification] = field(default_factory=list)
    total_regions: int = 0
    total_availability_zones: int = 0
    
    # Performance capabilities
    performance_capabilities: Optional[PerformanceCapability] = None
    
    # Support options
    support_tiers: List[SupportTier] = field(default_factory=list)
    
    # Additional metadata
    market_share_percentage: Optional[float] = None
    enterprise_customers: Optional[int] = None
    documentation_quality_score: Optional[float] = None
    community_size: Optional[int] = None
    
    def __repr__(self):
        return f"<CloudProvider(name='{self.provider_name.value}', services={len(self.services)})>"
    
    def get_service(self, service_id: str) -> Optional[ServiceSpecification]:
        """Get service by ID"""
        return self.services.get(service_id)
    
    def get_services_by_category(self, category: ServiceCategory) -> List[ServiceSpecification]:
        """Get all services in a category"""
        service_ids = self.service_categories.get(category, [])
        return [self.services[sid] for sid in service_ids if sid in self.services]
    
    def get_pricing(self, service_id: str, region: str) -> Optional[ServicePricing]:
        """Get pricing for a service in a region"""
        pricing_key = f"{service_id}_{region}"
        return self.pricing_data.get(pricing_key)
    
    def has_compliance(self, framework: ComplianceFramework) -> bool:
        """Check if provider has specific compliance certification"""
        return framework in self.compliance_frameworks
    
    def get_region(self, region_id: str) -> Optional[RegionSpecification]:
        """Get region specification by ID"""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None
    
    def supports_service_in_region(self, service_id: str, region_id: str) -> bool:
        """Check if a service is available in a specific region"""
        region = self.get_region(region_id)
        if not region:
            return False
        return service_id in region.services_available


@dataclass
class ServiceComparison:
    """Comparison of equivalent services across providers"""
    service_category: ServiceCategory
    service_purpose: str
    aws_service: Optional[ServiceSpecification] = None
    gcp_service: Optional[ServiceSpecification] = None
    azure_service: Optional[ServiceSpecification] = None
    feature_comparison: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pricing_comparison: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"<ServiceComparison(category='{self.service_category.value}', purpose='{self.service_purpose}')>"


@dataclass
class ProviderCatalog:
    """Central catalog of all cloud providers"""
    providers: Dict[CloudProviderName, CloudProvider] = field(default_factory=dict)
    service_comparisons: List[ServiceComparison] = field(default_factory=list)
    last_updated: Optional[str] = None
    
    def __repr__(self):
        return f"<ProviderCatalog(providers={len(self.providers)})>"
    
    def get_provider(self, provider_name: CloudProviderName) -> Optional[CloudProvider]:
        """Get provider by name"""
        return self.providers.get(provider_name)
    
    def add_provider(self, provider: CloudProvider) -> None:
        """Add or update a provider in the catalog"""
        self.providers[provider.provider_name] = provider
    
    def get_service_comparison(self, category: ServiceCategory, purpose: str) -> Optional[ServiceComparison]:
        """Get service comparison for a specific category and purpose"""
        for comparison in self.service_comparisons:
            if comparison.service_category == category and comparison.service_purpose == purpose:
                return comparison
        return None
    
    def compare_compliance(self, framework: ComplianceFramework) -> Dict[str, bool]:
        """Compare compliance support across all providers"""
        return {
            provider_name.value: provider.has_compliance(framework)
            for provider_name, provider in self.providers.items()
        }
