"""
Cloud Provider Pricing Catalog and Cost Estimation

This module provides pricing data for cloud services and utilities for
cost estimation and comparison across providers.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from .provider_catalog import (
    CloudProviderName, ServicePricing, PricingTier, PricingModel
)


class InstanceSize(Enum):
    """Standard instance size categories"""
    NANO = "nano"
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
    XXLARGE = "2xlarge"
    XXXLARGE = "4xlarge"


class StorageClass(Enum):
    """Storage class types"""
    STANDARD = "standard"
    INFREQUENT_ACCESS = "infrequent_access"
    ARCHIVE = "archive"
    INTELLIGENT = "intelligent"


def create_aws_pricing() -> Dict[str, ServicePricing]:
    """Create AWS pricing data"""
    pricing = {}
    
    # EC2 Pricing (us-east-1)
    pricing["ec2_us-east-1"] = ServicePricing(
        service_id="ec2",
        region="us-east-1",
        pricing_tiers=[
            PricingTier(
                tier_name="t3.micro",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0104")
            ),
            PricingTier(
                tier_name="t3.small",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0208")
            ),
            PricingTier(
                tier_name="t3.medium",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0416")
            ),
            PricingTier(
                tier_name="m5.large",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.096")
            ),
            PricingTier(
                tier_name="m5.xlarge",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.192")
            ),
            PricingTier(
                tier_name="m5.2xlarge",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.384")
            )
        ],
        free_tier={"hours_per_month": 750, "instance_type": "t2.micro"},
        currency="USD"
    )
    
    # S3 Pricing (us-east-1)
    pricing["s3_us-east-1"] = ServicePricing(
        service_id="s3",
        region="us-east-1",
        pricing_tiers=[
            PricingTier(
                tier_name="Standard",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.023")
            ),
            PricingTier(
                tier_name="Infrequent Access",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.0125")
            ),
            PricingTier(
                tier_name="Glacier",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.004")
            )
        ],
        additional_costs={
            "data_transfer_out": Decimal("0.09"),  # per GB
            "put_requests": Decimal("0.005"),  # per 1000 requests
            "get_requests": Decimal("0.0004")  # per 1000 requests
        },
        free_tier={"storage_gb": 5, "get_requests": 20000, "put_requests": 2000},
        currency="USD"
    )
    
    # RDS Pricing (us-east-1)
    pricing["rds_us-east-1"] = ServicePricing(
        service_id="rds",
        region="us-east-1",
        pricing_tiers=[
            PricingTier(
                tier_name="db.t3.micro",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.017")
            ),
            PricingTier(
                tier_name="db.t3.small",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.034")
            ),
            PricingTier(
                tier_name="db.m5.large",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.154")
            ),
            PricingTier(
                tier_name="db.m5.xlarge",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.308")
            )
        ],
        additional_costs={
            "storage_gp2": Decimal("0.115"),  # per GB/month
            "backup_storage": Decimal("0.095"),  # per GB/month
            "data_transfer_out": Decimal("0.09")  # per GB
        },
        free_tier={"hours_per_month": 750, "instance_type": "db.t2.micro", "storage_gb": 20},
        currency="USD"
    )
    
    # Lambda Pricing (us-east-1)
    pricing["lambda_us-east-1"] = ServicePricing(
        service_id="lambda",
        region="us-east-1",
        pricing_tiers=[
            PricingTier(
                tier_name="Requests",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per 1M requests",
                price_per_unit=Decimal("0.20")
            ),
            PricingTier(
                tier_name="Duration",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB-second",
                price_per_unit=Decimal("0.0000166667")
            )
        ],
        free_tier={"requests_per_month": 1000000, "compute_gb_seconds": 400000},
        currency="USD"
    )
    
    # DynamoDB Pricing (us-east-1)
    pricing["dynamodb_us-east-1"] = ServicePricing(
        service_id="dynamodb",
        region="us-east-1",
        pricing_tiers=[
            PricingTier(
                tier_name="On-Demand Write",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per million write request units",
                price_per_unit=Decimal("1.25")
            ),
            PricingTier(
                tier_name="On-Demand Read",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per million read request units",
                price_per_unit=Decimal("0.25")
            ),
            PricingTier(
                tier_name="Storage",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.25")
            )
        ],
        free_tier={"write_units": 25, "read_units": 25, "storage_gb": 25},
        currency="USD"
    )
    
    return pricing


def create_gcp_pricing() -> Dict[str, ServicePricing]:
    """Create GCP pricing data"""
    pricing = {}
    
    # Compute Engine Pricing (us-central1)
    pricing["compute_engine_us-central1"] = ServicePricing(
        service_id="compute_engine",
        region="us-central1",
        pricing_tiers=[
            PricingTier(
                tier_name="e2-micro",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0084")
            ),
            PricingTier(
                tier_name="e2-small",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0168")
            ),
            PricingTier(
                tier_name="e2-medium",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0336")
            ),
            PricingTier(
                tier_name="n2-standard-2",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0971")
            ),
            PricingTier(
                tier_name="n2-standard-4",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.1942")
            )
        ],
        free_tier={"hours_per_month": 730, "instance_type": "e2-micro"},
        currency="USD"
    )
    
    # Cloud Storage Pricing (us-central1)
    pricing["cloud_storage_us-central1"] = ServicePricing(
        service_id="cloud_storage",
        region="us-central1",
        pricing_tiers=[
            PricingTier(
                tier_name="Standard",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.020")
            ),
            PricingTier(
                tier_name="Nearline",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.010")
            ),
            PricingTier(
                tier_name="Coldline",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.004")
            ),
            PricingTier(
                tier_name="Archive",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.0012")
            )
        ],
        additional_costs={
            "data_transfer_out": Decimal("0.12"),  # per GB
            "class_a_operations": Decimal("0.05"),  # per 10,000 operations
            "class_b_operations": Decimal("0.004")  # per 10,000 operations
        },
        free_tier={"storage_gb": 5, "class_a_operations": 5000, "class_b_operations": 50000},
        currency="USD"
    )
    
    # Cloud SQL Pricing (us-central1)
    pricing["cloud_sql_us-central1"] = ServicePricing(
        service_id="cloud_sql",
        region="us-central1",
        pricing_tiers=[
            PricingTier(
                tier_name="db-f1-micro",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0150")
            ),
            PricingTier(
                tier_name="db-g1-small",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0500")
            ),
            PricingTier(
                tier_name="db-n1-standard-1",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0965")
            ),
            PricingTier(
                tier_name="db-n1-standard-2",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.1930")
            )
        ],
        additional_costs={
            "storage_ssd": Decimal("0.170"),  # per GB/month
            "backup_storage": Decimal("0.080"),  # per GB/month
            "data_transfer_out": Decimal("0.12")  # per GB
        },
        currency="USD"
    )
    
    # Cloud Functions Pricing (us-central1)
    pricing["cloud_functions_us-central1"] = ServicePricing(
        service_id="cloud_functions",
        region="us-central1",
        pricing_tiers=[
            PricingTier(
                tier_name="Invocations",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per 1M invocations",
                price_per_unit=Decimal("0.40")
            ),
            PricingTier(
                tier_name="Compute Time",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB-second",
                price_per_unit=Decimal("0.0000025")
            )
        ],
        free_tier={"invocations_per_month": 2000000, "compute_gb_seconds": 400000},
        currency="USD"
    )
    
    # BigQuery Pricing (us-central1)
    pricing["bigquery_us-central1"] = ServicePricing(
        service_id="bigquery",
        region="us-central1",
        pricing_tiers=[
            PricingTier(
                tier_name="On-Demand Analysis",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per TB processed",
                price_per_unit=Decimal("5.00")
            ),
            PricingTier(
                tier_name="Storage",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.020")
            ),
            PricingTier(
                tier_name="Long-term Storage",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.010")
            )
        ],
        free_tier={"analysis_tb": 1, "storage_gb": 10},
        currency="USD"
    )
    
    return pricing


def create_azure_pricing() -> Dict[str, ServicePricing]:
    """Create Azure pricing data"""
    pricing = {}
    
    # Virtual Machines Pricing (eastus)
    pricing["virtual_machines_eastus"] = ServicePricing(
        service_id="virtual_machines",
        region="eastus",
        pricing_tiers=[
            PricingTier(
                tier_name="B1s",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0104")
            ),
            PricingTier(
                tier_name="B2s",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.0416")
            ),
            PricingTier(
                tier_name="D2s_v3",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.096")
            ),
            PricingTier(
                tier_name="D4s_v3",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.192")
            ),
            PricingTier(
                tier_name="D8s_v3",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per hour",
                price_per_unit=Decimal("0.384")
            )
        ],
        free_tier={"hours_per_month": 750, "instance_type": "B1s"},
        currency="USD"
    )
    
    # Blob Storage Pricing (eastus)
    pricing["blob_storage_eastus"] = ServicePricing(
        service_id="blob_storage",
        region="eastus",
        pricing_tiers=[
            PricingTier(
                tier_name="Hot",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.0184")
            ),
            PricingTier(
                tier_name="Cool",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.0100")
            ),
            PricingTier(
                tier_name="Archive",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.00099")
            )
        ],
        additional_costs={
            "data_transfer_out": Decimal("0.087"),  # per GB
            "write_operations": Decimal("0.05"),  # per 10,000 operations
            "read_operations": Decimal("0.004")  # per 10,000 operations
        },
        free_tier={"storage_gb": 5, "write_operations": 10000, "read_operations": 50000},
        currency="USD"
    )
    
    # SQL Database Pricing (eastus)
    pricing["sql_database_eastus"] = ServicePricing(
        service_id="sql_database",
        region="eastus",
        pricing_tiers=[
            PricingTier(
                tier_name="Basic",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per day",
                price_per_unit=Decimal("0.0068")
            ),
            PricingTier(
                tier_name="S0",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per day",
                price_per_unit=Decimal("0.0203")
            ),
            PricingTier(
                tier_name="S1",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per day",
                price_per_unit=Decimal("0.0406")
            ),
            PricingTier(
                tier_name="S2",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per day",
                price_per_unit=Decimal("0.1015")
            )
        ],
        additional_costs={
            "backup_storage": Decimal("0.10"),  # per GB/month
            "data_transfer_out": Decimal("0.087")  # per GB
        },
        currency="USD"
    )
    
    # Azure Functions Pricing (eastus)
    pricing["azure_functions_eastus"] = ServicePricing(
        service_id="azure_functions",
        region="eastus",
        pricing_tiers=[
            PricingTier(
                tier_name="Executions",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per 1M executions",
                price_per_unit=Decimal("0.20")
            ),
            PricingTier(
                tier_name="Execution Time",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB-second",
                price_per_unit=Decimal("0.000016")
            )
        ],
        free_tier={"executions_per_month": 1000000, "compute_gb_seconds": 400000},
        currency="USD"
    )
    
    # Cosmos DB Pricing (eastus)
    pricing["cosmos_db_eastus"] = ServicePricing(
        service_id="cosmos_db",
        region="eastus",
        pricing_tiers=[
            PricingTier(
                tier_name="Provisioned Throughput",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per 100 RU/s per hour",
                price_per_unit=Decimal("0.008")
            ),
            PricingTier(
                tier_name="Storage",
                pricing_model=PricingModel.ON_DEMAND,
                unit="per GB/month",
                price_per_unit=Decimal("0.25")
            )
        ],
        free_tier={"throughput_rus": 400, "storage_gb": 5},
        currency="USD"
    )
    
    return pricing


@dataclass
class CostEstimate:
    """Cost estimation result"""
    provider: CloudProviderName
    service_id: str
    monthly_cost: Decimal
    annual_cost: Decimal
    breakdown: Dict[str, Decimal] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    confidence_level: str = "medium"  # low, medium, high


class CostEstimator:
    """Utility for estimating and comparing cloud costs"""
    
    def __init__(self):
        self.aws_pricing = create_aws_pricing()
        self.gcp_pricing = create_gcp_pricing()
        self.azure_pricing = create_azure_pricing()
        
        self.provider_pricing = {
            CloudProviderName.AWS: self.aws_pricing,
            CloudProviderName.GCP: self.gcp_pricing,
            CloudProviderName.AZURE: self.azure_pricing
        }
    
    def estimate_compute_cost(
        self,
        provider: CloudProviderName,
        instance_type: str,
        hours_per_month: int = 730,
        region: str = "us-east-1"
    ) -> CostEstimate:
        """Estimate compute instance cost"""
        service_map = {
            CloudProviderName.AWS: "ec2",
            CloudProviderName.GCP: "compute_engine",
            CloudProviderName.AZURE: "virtual_machines"
        }
        
        region_map = {
            CloudProviderName.AWS: "us-east-1",
            CloudProviderName.GCP: "us-central1",
            CloudProviderName.AZURE: "eastus"
        }
        
        service_id = service_map[provider]
        pricing_key = f"{service_id}_{region_map[provider]}"
        pricing_data = self.provider_pricing[provider].get(pricing_key)
        
        if not pricing_data:
            return CostEstimate(
                provider=provider,
                service_id=service_id,
                monthly_cost=Decimal("0"),
                annual_cost=Decimal("0"),
                assumptions=["Pricing data not available"]
            )
        
        # Find matching tier
        hourly_rate = Decimal("0")
        for tier in pricing_data.pricing_tiers:
            if instance_type.lower() in tier.tier_name.lower():
                hourly_rate = tier.price_per_unit
                break
        
        if hourly_rate == Decimal("0"):
            # Use first tier as default
            hourly_rate = pricing_data.pricing_tiers[0].price_per_unit if pricing_data.pricing_tiers else Decimal("0")
        
        monthly_cost = hourly_rate * Decimal(str(hours_per_month))
        annual_cost = monthly_cost * Decimal("12")
        
        return CostEstimate(
            provider=provider,
            service_id=service_id,
            monthly_cost=monthly_cost,
            annual_cost=annual_cost,
            breakdown={"compute": monthly_cost},
            assumptions=[f"Instance type: {instance_type}", f"Hours per month: {hours_per_month}"],
            confidence_level="high"
        )
    
    def estimate_storage_cost(
        self,
        provider: CloudProviderName,
        storage_gb: float,
        storage_class: str = "standard",
        region: str = "us-east-1"
    ) -> CostEstimate:
        """Estimate storage cost"""
        service_map = {
            CloudProviderName.AWS: "s3",
            CloudProviderName.GCP: "cloud_storage",
            CloudProviderName.AZURE: "blob_storage"
        }
        
        region_map = {
            CloudProviderName.AWS: "us-east-1",
            CloudProviderName.GCP: "us-central1",
            CloudProviderName.AZURE: "eastus"
        }
        
        service_id = service_map[provider]
        pricing_key = f"{service_id}_{region_map[provider]}"
        pricing_data = self.provider_pricing[provider].get(pricing_key)
        
        if not pricing_data:
            return CostEstimate(
                provider=provider,
                service_id=service_id,
                monthly_cost=Decimal("0"),
                annual_cost=Decimal("0"),
                assumptions=["Pricing data not available"]
            )
        
        # Find matching tier
        gb_rate = Decimal("0")
        for tier in pricing_data.pricing_tiers:
            if storage_class.lower() in tier.tier_name.lower():
                gb_rate = tier.price_per_unit
                break
        
        if gb_rate == Decimal("0"):
            gb_rate = pricing_data.pricing_tiers[0].price_per_unit if pricing_data.pricing_tiers else Decimal("0")
        
        monthly_cost = gb_rate * Decimal(str(storage_gb))
        annual_cost = monthly_cost * Decimal("12")
        
        return CostEstimate(
            provider=provider,
            service_id=service_id,
            monthly_cost=monthly_cost,
            annual_cost=annual_cost,
            breakdown={"storage": monthly_cost},
            assumptions=[f"Storage: {storage_gb} GB", f"Storage class: {storage_class}"],
            confidence_level="high"
        )
    
    def estimate_database_cost(
        self,
        provider: CloudProviderName,
        instance_type: str,
        storage_gb: float = 100,
        hours_per_month: int = 730,
        region: str = "us-east-1"
    ) -> CostEstimate:
        """Estimate database cost"""
        service_map = {
            CloudProviderName.AWS: "rds",
            CloudProviderName.GCP: "cloud_sql",
            CloudProviderName.AZURE: "sql_database"
        }
        
        region_map = {
            CloudProviderName.AWS: "us-east-1",
            CloudProviderName.GCP: "us-central1",
            CloudProviderName.AZURE: "eastus"
        }
        
        service_id = service_map[provider]
        pricing_key = f"{service_id}_{region_map[provider]}"
        pricing_data = self.provider_pricing[provider].get(pricing_key)
        
        if not pricing_data:
            return CostEstimate(
                provider=provider,
                service_id=service_id,
                monthly_cost=Decimal("0"),
                annual_cost=Decimal("0"),
                assumptions=["Pricing data not available"]
            )
        
        # Find matching tier
        hourly_rate = Decimal("0")
        for tier in pricing_data.pricing_tiers:
            if instance_type.lower() in tier.tier_name.lower():
                hourly_rate = tier.price_per_unit
                break
        
        if hourly_rate == Decimal("0"):
            hourly_rate = pricing_data.pricing_tiers[0].price_per_unit if pricing_data.pricing_tiers else Decimal("0")
        
        # Calculate instance cost
        if provider == CloudProviderName.AZURE:
            # Azure SQL Database is priced per day
            instance_cost = hourly_rate * Decimal("30")  # Approximate days per month
        else:
            instance_cost = hourly_rate * Decimal(str(hours_per_month))
        
        # Calculate storage cost
        storage_cost = Decimal("0")
        if "storage_gp2" in pricing_data.additional_costs:
            storage_cost = pricing_data.additional_costs["storage_gp2"] * Decimal(str(storage_gb))
        elif "storage_ssd" in pricing_data.additional_costs:
            storage_cost = pricing_data.additional_costs["storage_ssd"] * Decimal(str(storage_gb))
        
        monthly_cost = instance_cost + storage_cost
        annual_cost = monthly_cost * Decimal("12")
        
        return CostEstimate(
            provider=provider,
            service_id=service_id,
            monthly_cost=monthly_cost,
            annual_cost=annual_cost,
            breakdown={"instance": instance_cost, "storage": storage_cost},
            assumptions=[
                f"Instance type: {instance_type}",
                f"Storage: {storage_gb} GB",
                f"Hours per month: {hours_per_month}"
            ],
            confidence_level="high"
        )
    
    def compare_costs_across_providers(
        self,
        workload_type: str,
        **kwargs
    ) -> Dict[CloudProviderName, CostEstimate]:
        """Compare costs across all providers for a workload"""
        results = {}
        
        for provider in [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]:
            if workload_type == "compute":
                estimate = self.estimate_compute_cost(provider, **kwargs)
            elif workload_type == "storage":
                estimate = self.estimate_storage_cost(provider, **kwargs)
            elif workload_type == "database":
                estimate = self.estimate_database_cost(provider, **kwargs)
            else:
                continue
            
            results[provider] = estimate
        
        return results
    
    def get_cheapest_provider(
        self,
        workload_type: str,
        **kwargs
    ) -> Tuple[CloudProviderName, CostEstimate]:
        """Find the cheapest provider for a workload"""
        comparisons = self.compare_costs_across_providers(workload_type, **kwargs)
        
        if not comparisons:
            return None, None
        
        cheapest_provider = min(
            comparisons.items(),
            key=lambda x: x[1].monthly_cost
        )
        
        return cheapest_provider


# Global cost estimator instance
_cost_estimator_instance = None


def get_cost_estimator() -> CostEstimator:
    """Get or create the global cost estimator instance"""
    global _cost_estimator_instance
    if _cost_estimator_instance is None:
        _cost_estimator_instance = CostEstimator()
    return _cost_estimator_instance
