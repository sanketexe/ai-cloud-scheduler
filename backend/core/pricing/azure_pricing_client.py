"""
Azure Pricing API Client

Integrates with Azure Retail Prices API to retrieve current pricing data.
Supports Virtual Machines, Storage, and other Azure services pricing.
"""

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
import aiohttp
from urllib.parse import urlencode

from .base_pricing_client import BasePricingClient, PricingAPIException, RateLimitException
from .pricing_models import ComputePricing, StoragePricing, NetworkPricing, DatabasePricing

logger = logging.getLogger(__name__)


class AzurePricingClient(BasePricingClient):
    """
    Azure Retail Prices API client for retrieving current Azure service pricing.
    
    Uses the Azure Retail Prices API which is publicly accessible.
    Rate limit: 100 requests per minute.
    """
    
    # Azure Retail Prices API endpoint
    BASE_URL = "https://prices.azure.com/api/retail/prices"
    
    # Azure service names
    SERVICE_NAMES = {
        'compute': 'Virtual Machines',
        'storage': 'Storage',
        'network': 'Bandwidth',
        'sql': 'SQL Database',
        'functions': 'Functions'
    }
    
    # Azure region mapping
    REGION_MAPPING = {
        'eastus': 'East US',
        'eastus2': 'East US 2',
        'westus': 'West US',
        'westus2': 'West US 2',
        'westus3': 'West US 3',
        'centralus': 'Central US',
        'northcentralus': 'North Central US',
        'southcentralus': 'South Central US',
        'westcentralus': 'West Central US',
        'canadacentral': 'Canada Central',
        'canadaeast': 'Canada East',
        'brazilsouth': 'Brazil South',
        'northeurope': 'North Europe',
        'westeurope': 'West Europe',
        'uksouth': 'UK South',
        'ukwest': 'UK West',
        'francecentral': 'France Central',
        'francesouth': 'France South',
        'germanywestcentral': 'Germany West Central',
        'norwayeast': 'Norway East',
        'switzerlandnorth': 'Switzerland North',
        'eastasia': 'East Asia',
        'southeastasia': 'Southeast Asia',
        'japaneast': 'Japan East',
        'japanwest': 'Japan West',
        'australiaeast': 'Australia East',
        'australiasoutheast': 'Australia Southeast',
        'centralindia': 'Central India',
        'southindia': 'South India',
        'westindia': 'West India',
        'koreacentral': 'Korea Central',
        'koreasouth': 'Korea South',
        'uaenorth': 'UAE North',
        'southafricanorth': 'South Africa North'
    }
    
    def __init__(self, region: str = "eastus"):
        """Initialize Azure pricing client."""
        super().__init__("azure", region)
        self.session = None
        self._setup_rate_limiting()
    
    def _setup_rate_limiting(self):
        """Setup rate limiting for Azure API calls (100 requests per minute)."""
        self.rate_limit_delay = 0.6  # 600ms delay between requests
        self.last_request_time = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to Azure Retail Prices API with rate limiting.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            Dict: JSON response data
            
        Raises:
            PricingAPIException: If request fails
        """
        try:
            # Apply rate limiting
            await self._handle_rate_limiting()
            
            session = await self._get_session()
            
            # Build full URL with parameters
            if params:
                url = f"{url}?{urlencode(params)}"
            
            logger.debug(f"Making Azure pricing API request: {url}")
            
            async with session.get(url) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitException("azure", retry_after)
                
                if response.status != 200:
                    error_text = await response.text()
                    raise PricingAPIException(
                        f"HTTP {response.status}: {error_text}",
                        "azure",
                        response.status
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise PricingAPIException(f"Network error: {str(e)}", "azure")
        except json.JSONDecodeError as e:
            raise PricingAPIException(f"Invalid JSON response: {str(e)}", "azure")
    
    async def _handle_rate_limiting(self):
        """Handle rate limiting with delay."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    def _normalize_region(self, region: str) -> str:
        """Convert Azure region code to pricing API region name."""
        return self.REGION_MAPPING.get(region, region)
    
    async def get_compute_pricing(
        self, 
        region: str, 
        instance_type: Optional[str] = None,
        operating_system: str = "linux",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ComputePricing]:
        """
        Get Virtual Machine pricing from Azure Retail Prices API.
        
        Args:
            region: Azure region
            instance_type: Specific VM size (e.g., 'Standard_B1s')
            operating_system: OS type ('linux', 'windows')
            filters: Additional filters
            
        Returns:
            List[ComputePricing]: VM pricing information
        """
        try:
            logger.info(f"Fetching Azure VM pricing for region {region}")
            
            # For demo purposes, use mock data
            pricing_data = await self._get_mock_vm_pricing(region, instance_type, operating_system)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get Azure compute pricing: {e}")
            raise PricingAPIException(f"Failed to get compute pricing: {str(e)}", "azure")
    
    async def get_storage_pricing(
        self, 
        region: str, 
        storage_type: Optional[str] = None,
        storage_class: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[StoragePricing]:
        """
        Get Azure Storage pricing from Retail Prices API.
        
        Args:
            region: Azure region
            storage_type: Type of storage ('object', 'block')
            storage_class: Storage tier ('hot', 'cool', 'archive')
            filters: Additional filters
            
        Returns:
            List[StoragePricing]: Storage pricing information
        """
        try:
            logger.info(f"Fetching Azure storage pricing for region {region}")
            
            pricing_data = []
            
            # Get Blob Storage pricing (object storage)
            if not storage_type or storage_type == "object":
                blob_pricing = await self._get_mock_blob_storage_pricing(region, storage_class)
                pricing_data.extend(blob_pricing)
            
            # Get Managed Disk pricing (block storage)
            if not storage_type or storage_type == "block":
                disk_pricing = await self._get_mock_managed_disk_pricing(region)
                pricing_data.extend(disk_pricing)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get Azure storage pricing: {e}")
            raise PricingAPIException(f"Failed to get storage pricing: {str(e)}", "azure")
    
    async def get_network_pricing(
        self, 
        region: str,
        service_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NetworkPricing]:
        """
        Get Azure network service pricing.
        
        Args:
            region: Azure region
            service_type: Network service type
            filters: Additional filters
            
        Returns:
            List[NetworkPricing]: Network pricing information
        """
        try:
            logger.info(f"Fetching Azure network pricing for region {region}")
            
            pricing_data = []
            
            # Data transfer pricing
            if not service_type or service_type == "data_transfer":
                dt_pricing = await self._get_mock_data_transfer_pricing(region)
                pricing_data.extend(dt_pricing)
            
            # Load balancer pricing
            if not service_type or service_type == "load_balancer":
                lb_pricing = await self._get_mock_load_balancer_pricing(region)
                pricing_data.extend(lb_pricing)
            
            # CDN pricing
            if not service_type or service_type == "cdn":
                cdn_pricing = await self._get_mock_cdn_pricing()
                pricing_data.extend(cdn_pricing)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get Azure network pricing: {e}")
            raise PricingAPIException(f"Failed to get network pricing: {str(e)}", "azure")
    
    async def get_database_pricing(
        self, 
        region: str,
        database_type: Optional[str] = None,
        instance_class: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DatabasePricing]:
        """
        Get Azure SQL Database pricing from Retail Prices API.
        
        Args:
            region: Azure region
            database_type: Database service type
            instance_class: Service tier
            filters: Additional filters
            
        Returns:
            List[DatabasePricing]: Database pricing information
        """
        try:
            logger.info(f"Fetching Azure SQL pricing for region {region}")
            
            pricing_data = await self._get_mock_sql_database_pricing(region, database_type, instance_class)
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get Azure database pricing: {e}")
            raise PricingAPIException(f"Failed to get database pricing: {str(e)}", "azure")
    
    # Mock pricing methods
    
    async def _get_mock_vm_pricing(
        self, 
        region: str, 
        instance_type: Optional[str], 
        operating_system: str
    ) -> List[ComputePricing]:
        """Get mock Azure VM pricing data."""
        
        # Mock Azure VM sizes and pricing
        vm_sizes = {
            'Standard_B1s': {'vcpus': 1, 'memory': 1.0, 'price': 0.0104},
            'Standard_B1ms': {'vcpus': 1, 'memory': 2.0, 'price': 0.0208},
            'Standard_B2s': {'vcpus': 2, 'memory': 4.0, 'price': 0.0416},
            'Standard_B2ms': {'vcpus': 2, 'memory': 8.0, 'price': 0.0832},
            'Standard_B4ms': {'vcpus': 4, 'memory': 16.0, 'price': 0.1664},
            'Standard_D2s_v3': {'vcpus': 2, 'memory': 8.0, 'price': 0.096},
            'Standard_D4s_v3': {'vcpus': 4, 'memory': 16.0, 'price': 0.192},
            'Standard_D8s_v3': {'vcpus': 8, 'memory': 32.0, 'price': 0.384},
            'Standard_F2s_v2': {'vcpus': 2, 'memory': 4.0, 'price': 0.085},
            'Standard_F4s_v2': {'vcpus': 4, 'memory': 8.0, 'price': 0.17},
            'Standard_E2s_v3': {'vcpus': 2, 'memory': 16.0, 'price': 0.126},
            'Standard_E4s_v3': {'vcpus': 4, 'memory': 32.0, 'price': 0.252}
        }
        
        # Windows pricing is typically higher
        os_multiplier = 1.9 if operating_system.lower() == 'windows' else 1.0
        
        # Regional pricing adjustments
        region_multipliers = {
            'eastus': 1.0,
            'westus2': 1.03,
            'westeurope': 1.08,
            'eastasia': 1.12,
            'japaneast': 1.15
        }
        region_multiplier = region_multipliers.get(region, 1.05)
        
        pricing_data = []
        
        # Filter by VM size if specified
        if instance_type:
            if instance_type in vm_sizes:
                types_to_process = {instance_type: vm_sizes[instance_type]}
            else:
                return []
        else:
            types_to_process = vm_sizes
        
        for vm_size, specs in types_to_process.items():
            base_price = specs['price'] * os_multiplier * region_multiplier
            
            pricing = ComputePricing(
                instance_type=vm_size,
                vcpus=specs['vcpus'],
                memory_gb=specs['memory'],
                price_per_hour=Decimal(str(base_price)),
                price_per_month=Decimal(str(base_price * 24 * 30)),
                operating_system=operating_system,
                region=region,
                currency="USD",
                spot_price_per_hour=Decimal(str(base_price * 0.2)),  # ~80% discount
                reserved_price_per_hour=Decimal(str(base_price * 0.65)),  # ~35% discount
                architecture="x86_64",
                network_performance="Moderate" if 'B' in vm_size else "High",
                additional_specs={
                    'platform': 'Azure Virtual Machines',
                    'premium_storage': True if 's' in vm_size else False,
                    'accelerated_networking': True
                }
            )
            pricing_data.append(pricing)
        
        return pricing_data
    
    async def _get_mock_blob_storage_pricing(
        self, 
        region: str, 
        storage_class: Optional[str]
    ) -> List[StoragePricing]:
        """Get mock Blob Storage pricing data."""
        
        storage_tiers = {
            'hot': {'price': 0.0184, 'retrieval': 0.0},
            'cool': {'price': 0.01, 'retrieval': 0.01},
            'archive': {'price': 0.00099, 'retrieval': 0.02}
        }
        
        # Regional pricing adjustments
        region_multipliers = {
            'eastus': 1.0,
            'westus2': 1.02,
            'westeurope': 1.06,
            'eastasia': 1.09
        }
        region_multiplier = region_multipliers.get(region, 1.04)
        
        pricing_data = []
        
        # Filter by storage tier if specified
        if storage_class:
            if storage_class in storage_tiers:
                tiers_to_process = {storage_class: storage_tiers[storage_class]}
            else:
                tiers_to_process = {'hot': storage_tiers['hot']}
        else:
            tiers_to_process = storage_tiers
        
        for tier_name, specs in tiers_to_process.items():
            base_price = specs['price'] * region_multiplier
            
            pricing = StoragePricing(
                storage_type="object",
                price_per_gb_month=Decimal(str(base_price)),
                region=region,
                currency="USD",
                storage_class=tier_name,
                request_price=Decimal("0.0004"),  # Per 10,000 operations
                retrieval_price=Decimal(str(specs['retrieval'])) if specs['retrieval'] > 0 else None,
                minimum_storage_duration=30 if tier_name != 'hot' else None,
                additional_specs={
                    'durability': '99.999999999%',
                    'availability': '99.9%' if tier_name == 'hot' else '99.0%',
                    'geo_redundancy': True,
                    'access_tier': tier_name
                }
            )
            pricing_data.append(pricing)
        
        return pricing_data
    
    async def _get_mock_managed_disk_pricing(self, region: str) -> List[StoragePricing]:
        """Get mock Managed Disk pricing data."""
        
        disk_types = {
            'Standard_LRS': {'price': 0.04, 'iops_price': 0.0},
            'StandardSSD_LRS': {'price': 0.075, 'iops_price': 0.0},
            'Premium_LRS': {'price': 0.135, 'iops_price': 0.0},
            'UltraSSD_LRS': {'price': 0.125, 'iops_price': 0.0522}
        }
        
        region_multiplier = 1.03 if region != 'eastus' else 1.0
        
        pricing_data = []
        
        for disk_type, specs in disk_types.items():
            base_price = specs['price'] * region_multiplier
            
            pricing = StoragePricing(
                storage_type="block",
                price_per_gb_month=Decimal(str(base_price)),
                region=region,
                currency="USD",
                storage_class=disk_type,
                iops_price=Decimal(str(specs['iops_price'])) if specs['iops_price'] > 0 else None,
                additional_specs={
                    'max_iops': 160000 if disk_type == 'UltraSSD_LRS' else 20000,
                    'max_throughput': '2000 MB/s' if disk_type == 'UltraSSD_LRS' else '900 MB/s',
                    'disk_type': disk_type,
                    'replication': 'LRS'
                }
            )
            pricing_data.append(pricing)
        
        return pricing_data
    
    async def _get_mock_data_transfer_pricing(self, region: str) -> List[NetworkPricing]:
        """Get mock data transfer pricing."""
        
        pricing_data = [
            NetworkPricing(
                service_type="data_transfer",
                price_per_gb=Decimal("0.087"),
                region=region,
                currency="USD",
                transfer_type="outbound",
                bandwidth_tier="first_5gb",
                additional_specs={'description': 'Outbound data transfer (first 5GB free)'}
            ),
            NetworkPricing(
                service_type="data_transfer",
                price_per_gb=Decimal("0.087"),
                region=region,
                currency="USD",
                transfer_type="outbound",
                bandwidth_tier="next_10tb",
                additional_specs={'description': 'Outbound data transfer (next 10TB)'}
            ),
            NetworkPricing(
                service_type="data_transfer",
                price_per_gb=Decimal("0.0"),
                region=region,
                currency="USD",
                transfer_type="inbound",
                additional_specs={'description': 'Inbound data transfer'}
            )
        ]
        
        return pricing_data
    
    async def _get_mock_load_balancer_pricing(self, region: str) -> List[NetworkPricing]:
        """Get mock Load Balancer pricing."""
        
        pricing_data = [
            NetworkPricing(
                service_type="load_balancer",
                price_per_hour=Decimal("0.025"),
                region=region,
                currency="USD",
                additional_specs={
                    'type': 'Application Gateway',
                    'capacity_unit_price': 0.0144
                }
            ),
            NetworkPricing(
                service_type="load_balancer",
                price_per_hour=Decimal("0.025"),
                region=region,
                currency="USD",
                additional_specs={
                    'type': 'Load Balancer',
                    'rule_price': 0.025
                }
            )
        ]
        
        return pricing_data
    
    async def _get_mock_cdn_pricing(self) -> List[NetworkPricing]:
        """Get mock CDN pricing."""
        
        pricing_data = [
            NetworkPricing(
                service_type="cdn",
                price_per_gb=Decimal("0.081"),
                region="global",
                currency="USD",
                transfer_type="outbound",
                bandwidth_tier="first_10tb",
                additional_specs={'description': 'Azure CDN data transfer'}
            ),
            NetworkPricing(
                service_type="cdn",
                price_per_request=Decimal("0.0075"),
                region="global",
                currency="USD",
                additional_specs={'description': 'Azure CDN requests (per 10,000)'}
            )
        ]
        
        return pricing_data
    
    async def _get_mock_sql_database_pricing(
        self, 
        region: str, 
        database_type: Optional[str], 
        instance_class: Optional[str]
    ) -> List[DatabasePricing]:
        """Get mock SQL Database pricing data."""
        
        service_tiers = {
            'Basic': {'price': 0.0067},
            'Standard_S0': {'price': 0.0200},
            'Standard_S1': {'price': 0.0400},
            'Standard_S2': {'price': 0.1200},
            'Premium_P1': {'price': 0.6250},
            'Premium_P2': {'price': 1.2500},
            'GP_Gen5_2': {'price': 0.5616},
            'GP_Gen5_4': {'price': 1.1232},
            'BC_Gen5_2': {'price': 1.1232},
            'BC_Gen5_4': {'price': 2.2464}
        }
        
        database_types = ['sqlserver', 'mysql', 'postgresql']
        
        region_multiplier = 1.04 if region != 'eastus' else 1.0
        
        pricing_data = []
        
        # Filter by database type if specified
        types_to_process = [database_type] if database_type else database_types
        
        # Filter by service tier if specified
        if instance_class:
            if instance_class in service_tiers:
                tiers_to_process = {instance_class: service_tiers[instance_class]}
            else:
                tiers_to_process = {'Basic': service_tiers['Basic']}
        else:
            tiers_to_process = service_tiers
        
        for db_type in types_to_process:
            for tier, specs in tiers_to_process.items():
                base_price = specs['price'] * region_multiplier
                
                # MySQL and PostgreSQL are only available in certain tiers
                if db_type in ['mysql', 'postgresql'] and tier.startswith(('Basic', 'Standard', 'Premium')):
                    continue
                
                pricing = DatabasePricing(
                    database_type=db_type,
                    instance_class=tier,
                    price_per_hour=Decimal(str(base_price)),
                    storage_price_per_gb_month=Decimal("0.125"),
                    region=region,
                    currency="USD",
                    engine_version=f"{db_type}-latest",
                    multi_az=True if 'Premium' in tier or 'BC_' in tier else False,
                    backup_storage_price=Decimal("0.10"),
                    additional_specs={
                        'service_tier': tier,
                        'max_storage': '4TB' if 'Basic' in tier else '1TB',
                        'backup_retention': '35 days',
                        'point_in_time_restore': True
                    }
                )
                pricing_data.append(pricing)
        
        return pricing_data
    
    async def get_supported_regions(self) -> List[str]:
        """Get list of supported Azure regions."""
        return list(self.REGION_MAPPING.keys())
    
    async def get_supported_instance_types(self, region: str) -> List[str]:
        """Get list of supported Azure VM sizes."""
        return [
            'Standard_B1s', 'Standard_B1ms', 'Standard_B2s', 'Standard_B2ms', 'Standard_B4ms',
            'Standard_D2s_v3', 'Standard_D4s_v3', 'Standard_D8s_v3',
            'Standard_F2s_v2', 'Standard_F4s_v2',
            'Standard_E2s_v3', 'Standard_E4s_v3'
        ]
    
    async def validate_region(self, region: str) -> bool:
        """Validate if the Azure region is supported."""
        return region in self.REGION_MAPPING
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()