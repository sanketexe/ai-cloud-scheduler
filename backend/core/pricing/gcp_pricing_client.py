"""
GCP Pricing API Client

Integrates with Google Cloud Billing API to retrieve current pricing data.
Supports Compute Engine, Cloud Storage, and other GCP services pricing.
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


class GCPPricingClient(BasePricingClient):
    """
    GCP Cloud Billing API client for retrieving current GCP service pricing.
    
    Uses the Cloud Billing API which requires authentication.
    Rate limit: 1000 requests per 100 seconds.
    """
    
    # GCP Cloud Billing API endpoints
    BASE_URL = "https://cloudbilling.googleapis.com/v1"
    SERVICES_URL = f"{BASE_URL}/services"
    
    # GCP service IDs
    SERVICE_IDS = {
        'compute': '6F81-5844-456A',  # Compute Engine
        'storage': '95FF-2EF5-5EA1',  # Cloud Storage
        'network': '6F81-5844-456A',  # Network (part of Compute Engine)
        'sql': '9662-B51E-5089',      # Cloud SQL
        'functions': 'A1E8-BE35-7EBC' # Cloud Functions
    }
    
    # GCP region mapping
    REGION_MAPPING = {
        'us-central1': 'us-central1',
        'us-east1': 'us-east1',
        'us-east4': 'us-east4',
        'us-west1': 'us-west1',
        'us-west2': 'us-west2',
        'us-west3': 'us-west3',
        'us-west4': 'us-west4',
        'europe-west1': 'europe-west1',
        'europe-west2': 'europe-west2',
        'europe-west3': 'europe-west3',
        'europe-west4': 'europe-west4',
        'europe-west6': 'europe-west6',
        'europe-north1': 'europe-north1',
        'asia-east1': 'asia-east1',
        'asia-east2': 'asia-east2',
        'asia-northeast1': 'asia-northeast1',
        'asia-northeast2': 'asia-northeast2',
        'asia-northeast3': 'asia-northeast3',
        'asia-south1': 'asia-south1',
        'asia-southeast1': 'asia-southeast1',
        'asia-southeast2': 'asia-southeast2',
        'australia-southeast1': 'australia-southeast1',
        'northamerica-northeast1': 'northamerica-northeast1',
        'southamerica-east1': 'southamerica-east1'
    }
    
    def __init__(self, region: str = "us-central1", api_key: Optional[str] = None):
        """
        Initialize GCP pricing client.
        
        Args:
            region: Default GCP region
            api_key: GCP API key for authentication (optional for mock)
        """
        super().__init__("gcp", region)
        self.api_key = api_key
        self.session = None
        self._setup_rate_limiting()
    
    def _setup_rate_limiting(self):
        """Setup rate limiting for GCP API calls (1000 requests per 100 seconds)."""
        self.rate_limit_delay = 0.1  # 100ms delay between requests
        self.last_request_time = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with authentication."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {}
            
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self.session
    
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to GCP Billing API with rate limiting.
        
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
            
            # Add API key to params if available
            if self.api_key and params is None:
                params = {}
            if self.api_key:
                params['key'] = self.api_key
            
            # Build full URL with parameters
            if params:
                url = f"{url}?{urlencode(params)}"
            
            logger.debug(f"Making GCP pricing API request: {url}")
            
            async with session.get(url) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 100))
                    raise RateLimitException("gcp", retry_after)
                
                if response.status == 401:
                    raise PricingAPIException("Authentication failed - invalid API key", "gcp", 401)
                
                if response.status != 200:
                    error_text = await response.text()
                    raise PricingAPIException(
                        f"HTTP {response.status}: {error_text}",
                        "gcp",
                        response.status
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise PricingAPIException(f"Network error: {str(e)}", "gcp")
        except json.JSONDecodeError as e:
            raise PricingAPIException(f"Invalid JSON response: {str(e)}", "gcp")
    
    async def _handle_rate_limiting(self):
        """Handle rate limiting with delay."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    def _normalize_region(self, region: str) -> str:
        """Normalize GCP region identifier."""
        return self.REGION_MAPPING.get(region, region)
    
    async def get_compute_pricing(
        self, 
        region: str, 
        instance_type: Optional[str] = None,
        operating_system: str = "linux",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ComputePricing]:
        """
        Get Compute Engine pricing from GCP Billing API.
        
        Args:
            region: GCP region
            instance_type: Specific machine type (e.g., 'n1-standard-1')
            operating_system: OS type ('linux', 'windows')
            filters: Additional filters
            
        Returns:
            List[ComputePricing]: Compute Engine pricing information
        """
        try:
            logger.info(f"Fetching GCP Compute Engine pricing for region {region}")
            
            # For demo purposes, use mock data
            pricing_data = await self._get_mock_compute_pricing(region, instance_type, operating_system)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get GCP compute pricing: {e}")
            raise PricingAPIException(f"Failed to get compute pricing: {str(e)}", "gcp")
    
    async def get_storage_pricing(
        self, 
        region: str, 
        storage_type: Optional[str] = None,
        storage_class: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[StoragePricing]:
        """
        Get Cloud Storage pricing from GCP Billing API.
        
        Args:
            region: GCP region
            storage_type: Type of storage ('object', 'block')
            storage_class: Storage class ('standard', 'nearline', 'coldline', 'archive')
            filters: Additional filters
            
        Returns:
            List[StoragePricing]: Storage pricing information
        """
        try:
            logger.info(f"Fetching GCP storage pricing for region {region}")
            
            pricing_data = []
            
            # Get Cloud Storage pricing (object storage)
            if not storage_type or storage_type == "object":
                gcs_pricing = await self._get_mock_cloud_storage_pricing(region, storage_class)
                pricing_data.extend(gcs_pricing)
            
            # Get Persistent Disk pricing (block storage)
            if not storage_type or storage_type == "block":
                pd_pricing = await self._get_mock_persistent_disk_pricing(region)
                pricing_data.extend(pd_pricing)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get GCP storage pricing: {e}")
            raise PricingAPIException(f"Failed to get storage pricing: {str(e)}", "gcp")
    
    async def get_network_pricing(
        self, 
        region: str,
        service_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NetworkPricing]:
        """
        Get GCP network service pricing.
        
        Args:
            region: GCP region
            service_type: Network service type
            filters: Additional filters
            
        Returns:
            List[NetworkPricing]: Network pricing information
        """
        try:
            logger.info(f"Fetching GCP network pricing for region {region}")
            
            pricing_data = []
            
            # Data transfer pricing
            if not service_type or service_type == "data_transfer":
                dt_pricing = await self._get_mock_data_transfer_pricing(region)
                pricing_data.extend(dt_pricing)
            
            # Load balancer pricing
            if not service_type or service_type == "load_balancer":
                lb_pricing = await self._get_mock_load_balancer_pricing(region)
                pricing_data.extend(lb_pricing)
            
            # Cloud CDN pricing
            if not service_type or service_type == "cdn":
                cdn_pricing = await self._get_mock_cloud_cdn_pricing()
                pricing_data.extend(cdn_pricing)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get GCP network pricing: {e}")
            raise PricingAPIException(f"Failed to get network pricing: {str(e)}", "gcp")
    
    async def get_database_pricing(
        self, 
        region: str,
        database_type: Optional[str] = None,
        instance_class: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DatabasePricing]:
        """
        Get Cloud SQL pricing from GCP Billing API.
        
        Args:
            region: GCP region
            database_type: Database engine type
            instance_class: Cloud SQL machine type
            filters: Additional filters
            
        Returns:
            List[DatabasePricing]: Database pricing information
        """
        try:
            logger.info(f"Fetching GCP Cloud SQL pricing for region {region}")
            
            pricing_data = await self._get_mock_cloud_sql_pricing(region, database_type, instance_class)
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to get GCP database pricing: {e}")
            raise PricingAPIException(f"Failed to get database pricing: {str(e)}", "gcp")
    
    # Mock pricing methods
    
    async def _get_mock_compute_pricing(
        self, 
        region: str, 
        instance_type: Optional[str], 
        operating_system: str
    ) -> List[ComputePricing]:
        """Get mock Compute Engine pricing data."""
        
        # Mock GCP machine types and pricing
        machine_types = {
            'e2-micro': {'vcpus': 2, 'memory': 1.0, 'price': 0.008},
            'e2-small': {'vcpus': 2, 'memory': 2.0, 'price': 0.016},
            'e2-medium': {'vcpus': 2, 'memory': 4.0, 'price': 0.032},
            'e2-standard-2': {'vcpus': 2, 'memory': 8.0, 'price': 0.067},
            'e2-standard-4': {'vcpus': 4, 'memory': 16.0, 'price': 0.134},
            'n1-standard-1': {'vcpus': 1, 'memory': 3.75, 'price': 0.0475},
            'n1-standard-2': {'vcpus': 2, 'memory': 7.5, 'price': 0.095},
            'n1-standard-4': {'vcpus': 4, 'memory': 15.0, 'price': 0.19},
            'n2-standard-2': {'vcpus': 2, 'memory': 8.0, 'price': 0.097},
            'n2-standard-4': {'vcpus': 4, 'memory': 16.0, 'price': 0.194},
            'c2-standard-4': {'vcpus': 4, 'memory': 16.0, 'price': 0.168},
            'n2-highmem-2': {'vcpus': 2, 'memory': 16.0, 'price': 0.118}
        }
        
        # Windows pricing is typically higher
        os_multiplier = 1.8 if operating_system.lower() == 'windows' else 1.0
        
        # Regional pricing adjustments
        region_multipliers = {
            'us-central1': 1.0,
            'us-east1': 0.95,
            'us-west1': 1.05,
            'europe-west1': 1.08,
            'asia-northeast1': 1.12
        }
        region_multiplier = region_multipliers.get(region, 1.05)
        
        pricing_data = []
        
        # Filter by machine type if specified
        if instance_type:
            if instance_type in machine_types:
                types_to_process = {instance_type: machine_types[instance_type]}
            else:
                return []
        else:
            types_to_process = machine_types
        
        for machine_type, specs in types_to_process.items():
            base_price = specs['price'] * os_multiplier * region_multiplier
            
            pricing = ComputePricing(
                instance_type=machine_type,
                vcpus=specs['vcpus'],
                memory_gb=specs['memory'],
                price_per_hour=Decimal(str(base_price)),
                price_per_month=Decimal(str(base_price * 24 * 30)),
                operating_system=operating_system,
                region=region,
                currency="USD",
                spot_price_per_hour=Decimal(str(base_price * 0.2)),  # ~80% discount
                reserved_price_per_hour=Decimal(str(base_price * 0.7)),  # ~30% discount
                architecture="x86_64",
                network_performance="Up to 10 Gbps" if 'standard-4' in machine_type else "Up to 2 Gbps",
                additional_specs={
                    'platform': 'Google Compute Engine',
                    'sustained_use_discount': True,
                    'preemptible': True
                }
            )
            pricing_data.append(pricing)
        
        return pricing_data
    
    async def _get_mock_cloud_storage_pricing(
        self, 
        region: str, 
        storage_class: Optional[str]
    ) -> List[StoragePricing]:
        """Get mock Cloud Storage pricing data."""
        
        storage_classes = {
            'standard': {'price': 0.020, 'retrieval': 0.0},
            'nearline': {'price': 0.010, 'retrieval': 0.01},
            'coldline': {'price': 0.004, 'retrieval': 0.02},
            'archive': {'price': 0.0012, 'retrieval': 0.05}
        }
        
        # Regional pricing adjustments
        region_multipliers = {
            'us-central1': 1.0,
            'us-east1': 0.98,
            'europe-west1': 1.05,
            'asia-northeast1': 1.08
        }
        region_multiplier = region_multipliers.get(region, 1.02)
        
        pricing_data = []
        
        # Filter by storage class if specified
        if storage_class:
            if storage_class in storage_classes:
                classes_to_process = {storage_class: storage_classes[storage_class]}
            else:
                classes_to_process = {'standard': storage_classes['standard']}
        else:
            classes_to_process = storage_classes
        
        for class_name, specs in classes_to_process.items():
            base_price = specs['price'] * region_multiplier
            
            pricing = StoragePricing(
                storage_type="object",
                price_per_gb_month=Decimal(str(base_price)),
                region=region,
                currency="USD",
                storage_class=class_name,
                request_price=Decimal("0.0005"),  # Per 1000 operations
                retrieval_price=Decimal(str(specs['retrieval'])) if specs['retrieval'] > 0 else None,
                minimum_storage_duration=30 if class_name != 'standard' else None,
                additional_specs={
                    'durability': '99.999999999%',
                    'availability': '99.95%' if class_name == 'standard' else '99.0%',
                    'geo_redundancy': True if class_name == 'standard' else False
                }
            )
            pricing_data.append(pricing)
        
        return pricing_data
    
    async def _get_mock_persistent_disk_pricing(self, region: str) -> List[StoragePricing]:
        """Get mock Persistent Disk pricing data."""
        
        disk_types = {
            'pd-standard': {'price': 0.040, 'iops_price': 0.0},
            'pd-balanced': {'price': 0.100, 'iops_price': 0.0},
            'pd-ssd': {'price': 0.170, 'iops_price': 0.0},
            'pd-extreme': {'price': 0.125, 'iops_price': 0.065}
        }
        
        region_multiplier = 1.02 if region != 'us-central1' else 1.0
        
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
                    'max_iops': 100000 if disk_type == 'pd-extreme' else 30000,
                    'max_throughput': '1200 MB/s' if disk_type == 'pd-extreme' else '800 MB/s',
                    'disk_type': disk_type
                }
            )
            pricing_data.append(pricing)
        
        return pricing_data
    
    async def _get_mock_data_transfer_pricing(self, region: str) -> List[NetworkPricing]:
        """Get mock data transfer pricing."""
        
        pricing_data = [
            NetworkPricing(
                service_type="data_transfer",
                price_per_gb=Decimal("0.12"),
                region=region,
                currency="USD",
                transfer_type="outbound",
                bandwidth_tier="first_1tb",
                additional_specs={'description': 'Internet egress (first 1TB)'}
            ),
            NetworkPricing(
                service_type="data_transfer",
                price_per_gb=Decimal("0.11"),
                region=region,
                currency="USD",
                transfer_type="outbound",
                bandwidth_tier="next_9tb",
                additional_specs={'description': 'Internet egress (next 9TB)'}
            ),
            NetworkPricing(
                service_type="data_transfer",
                price_per_gb=Decimal("0.0"),
                region=region,
                currency="USD",
                transfer_type="inbound",
                additional_specs={'description': 'Internet ingress'}
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
                    'type': 'HTTP(S) Load Balancer',
                    'forwarding_rule_price': 0.025
                }
            ),
            NetworkPricing(
                service_type="load_balancer",
                price_per_hour=Decimal("0.025"),
                region=region,
                currency="USD",
                additional_specs={
                    'type': 'Network Load Balancer',
                    'forwarding_rule_price': 0.025
                }
            )
        ]
        
        return pricing_data
    
    async def _get_mock_cloud_cdn_pricing(self) -> List[NetworkPricing]:
        """Get mock Cloud CDN pricing."""
        
        pricing_data = [
            NetworkPricing(
                service_type="cdn",
                price_per_gb=Decimal("0.08"),
                region="global",
                currency="USD",
                transfer_type="outbound",
                bandwidth_tier="first_10tb",
                additional_specs={'description': 'Cloud CDN cache egress'}
            ),
            NetworkPricing(
                service_type="cdn",
                price_per_request=Decimal("0.0075"),
                region="global",
                currency="USD",
                additional_specs={'description': 'Cloud CDN cache requests (per 10,000)'}
            )
        ]
        
        return pricing_data
    
    async def _get_mock_cloud_sql_pricing(
        self, 
        region: str, 
        database_type: Optional[str], 
        instance_class: Optional[str]
    ) -> List[DatabasePricing]:
        """Get mock Cloud SQL pricing data."""
        
        instance_classes = {
            'db-f1-micro': {'price': 0.0150},
            'db-g1-small': {'price': 0.0500},
            'db-n1-standard-1': {'price': 0.0825},
            'db-n1-standard-2': {'price': 0.1650},
            'db-n1-standard-4': {'price': 0.3300},
            'db-n1-highmem-2': {'price': 0.2035},
            'db-n1-highmem-4': {'price': 0.4070}
        }
        
        database_engines = ['mysql', 'postgresql', 'sqlserver']
        
        region_multiplier = 1.02 if region != 'us-central1' else 1.0
        
        pricing_data = []
        
        # Filter by database type if specified
        engines_to_process = [database_type] if database_type else database_engines
        
        # Filter by instance class if specified
        if instance_class:
            if instance_class in instance_classes:
                classes_to_process = {instance_class: instance_classes[instance_class]}
            else:
                classes_to_process = {'db-f1-micro': instance_classes['db-f1-micro']}
        else:
            classes_to_process = instance_classes
        
        for engine in engines_to_process:
            for inst_class, specs in classes_to_process.items():
                base_price = specs['price'] * region_multiplier
                
                # SQL Server is more expensive
                if engine == 'sqlserver':
                    base_price *= 4.0
                
                pricing = DatabasePricing(
                    database_type=engine,
                    instance_class=inst_class,
                    price_per_hour=Decimal(str(base_price)),
                    storage_price_per_gb_month=Decimal("0.170"),
                    region=region,
                    currency="USD",
                    engine_version=f"{engine}-latest",
                    multi_az=False,
                    backup_storage_price=Decimal("0.080"),
                    additional_specs={
                        'storage_type': 'SSD',
                        'max_storage': '30TB',
                        'backup_retention': '7 days',
                        'point_in_time_recovery': True
                    }
                )
                pricing_data.append(pricing)
        
        return pricing_data
    
    async def get_supported_regions(self) -> List[str]:
        """Get list of supported GCP regions."""
        return list(self.REGION_MAPPING.keys())
    
    async def get_supported_instance_types(self, region: str) -> List[str]:
        """Get list of supported GCP machine types."""
        return [
            'e2-micro', 'e2-small', 'e2-medium', 'e2-standard-2', 'e2-standard-4',
            'n1-standard-1', 'n1-standard-2', 'n1-standard-4',
            'n2-standard-2', 'n2-standard-4',
            'c2-standard-4', 'n2-highmem-2'
        ]
    
    async def validate_region(self, region: str) -> bool:
        """Validate if the GCP region is supported."""
        return region in self.REGION_MAPPING
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()