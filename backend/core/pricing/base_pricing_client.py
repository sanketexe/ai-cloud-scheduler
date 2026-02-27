"""
Base pricing client interface for multi-cloud cost comparison.

Defines the common interface that all cloud provider pricing clients must implement.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any

from .pricing_models import (
    ComputePricing, StoragePricing, NetworkPricing, DatabasePricing,
    PricingData, PricingQuery, PricingResponse, ServiceCategory
)

logger = logging.getLogger(__name__)


class BasePricingClient(ABC):
    """
    Abstract base class for cloud provider pricing clients.
    
    All provider-specific pricing clients must inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, provider_name: str, region: str = "us-east-1"):
        """
        Initialize the pricing client.
        
        Args:
            provider_name: Name of the cloud provider (aws, gcp, azure)
            region: Default region for pricing queries
        """
        self.provider_name = provider_name
        self.default_region = region
        self.session = None
        self.rate_limiter = None
        self._setup_rate_limiting()
        
        logger.info(f"Initialized {provider_name} pricing client for region {region}")
    
    def _setup_rate_limiting(self):
        """Setup rate limiting for API calls."""
        # This will be implemented by subclasses based on provider limits
        pass
    
    @abstractmethod
    async def get_compute_pricing(
        self, 
        region: str, 
        instance_type: Optional[str] = None,
        operating_system: str = "linux",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ComputePricing]:
        """
        Get compute instance pricing for the specified region.
        
        Args:
            region: AWS region (e.g., 'us-east-1')
            instance_type: Specific instance type to query (optional)
            operating_system: Operating system ('linux', 'windows')
            filters: Additional filters for the query
            
        Returns:
            List[ComputePricing]: List of compute pricing information
            
        Raises:
            PricingAPIException: If the API call fails
        """
        pass
    
    @abstractmethod
    async def get_storage_pricing(
        self, 
        region: str, 
        storage_type: Optional[str] = None,
        storage_class: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[StoragePricing]:
        """
        Get storage pricing for the specified region.
        
        Args:
            region: Region identifier
            storage_type: Type of storage (object, block, file)
            storage_class: Storage class (standard, infrequent, archive)
            filters: Additional filters for the query
            
        Returns:
            List[StoragePricing]: List of storage pricing information
            
        Raises:
            PricingAPIException: If the API call fails
        """
        pass
    
    @abstractmethod
    async def get_network_pricing(
        self, 
        region: str,
        service_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NetworkPricing]:
        """
        Get network service pricing for the specified region.
        
        Args:
            region: Region identifier
            service_type: Type of network service (data_transfer, load_balancer, cdn)
            filters: Additional filters for the query
            
        Returns:
            List[NetworkPricing]: List of network pricing information
            
        Raises:
            PricingAPIException: If the API call fails
        """
        pass
    
    async def get_database_pricing(
        self, 
        region: str,
        database_type: Optional[str] = None,
        instance_class: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DatabasePricing]:
        """
        Get database service pricing for the specified region.
        
        Args:
            region: Region identifier
            database_type: Type of database (mysql, postgresql, mongodb)
            instance_class: Database instance class
            filters: Additional filters for the query
            
        Returns:
            List[DatabasePricing]: List of database pricing information
            
        Raises:
            PricingAPIException: If the API call fails
        """
        # Default implementation returns empty list
        # Subclasses can override if they support database pricing
        return []
    
    async def get_comprehensive_pricing(
        self, 
        query: PricingQuery
    ) -> PricingResponse:
        """
        Get comprehensive pricing data for a service.
        
        Args:
            query: Pricing query parameters
            
        Returns:
            PricingResponse: Complete pricing response with all service types
        """
        try:
            start_time = datetime.utcnow()
            
            # Gather pricing data for all service types in parallel
            tasks = []
            
            # Always get compute pricing
            tasks.append(self.get_compute_pricing(
                query.region, 
                query.instance_type, 
                query.filters.get('operating_system', 'linux'),
                query.filters
            ))
            
            # Always get storage pricing
            tasks.append(self.get_storage_pricing(
                query.region,
                query.storage_type,
                query.filters.get('storage_class'),
                query.filters
            ))
            
            # Always get network pricing
            tasks.append(self.get_network_pricing(
                query.region,
                query.filters.get('network_service_type'),
                query.filters
            ))
            
            # Get database pricing if requested
            if query.filters and query.filters.get('include_database', False):
                tasks.append(self.get_database_pricing(
                    query.region,
                    query.filters.get('database_type'),
                    query.filters.get('database_instance_class'),
                    query.filters
                ))
            else:
                tasks.append(asyncio.create_task(self._empty_database_pricing()))
            
            # Execute all pricing queries in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            compute_pricing = results[0] if not isinstance(results[0], Exception) else []
            storage_pricing = results[1] if not isinstance(results[1], Exception) else []
            network_pricing = results[2] if not isinstance(results[2], Exception) else []
            database_pricing = results[3] if not isinstance(results[3], Exception) else []
            
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    service_types = ['compute', 'storage', 'network', 'database']
                    logger.warning(f"Failed to get {service_types[i]} pricing: {result}")
            
            # Create pricing data
            pricing_data = PricingData(
                provider=self.provider_name,
                service_name=query.service_name,
                service_category=ServiceCategory.COMPUTE,  # Default category
                region=query.region,
                pricing_date=datetime.utcnow(),
                compute_pricing=compute_pricing,
                storage_pricing=storage_pricing,
                network_pricing=network_pricing,
                database_pricing=database_pricing
            )
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return PricingResponse(
                success=True,
                data=pricing_data,
                response_time_ms=response_time_ms,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive pricing for {query.provider}: {e}")
            return PricingResponse(
                success=False,
                error_message=str(e)
            )
    
    async def _empty_database_pricing(self) -> List[DatabasePricing]:
        """Return empty database pricing list."""
        return []
    
    async def validate_region(self, region: str) -> bool:
        """
        Validate if the region is supported by the provider.
        
        Args:
            region: Region identifier to validate
            
        Returns:
            bool: True if region is supported, False otherwise
        """
        # Default implementation - subclasses should override
        return True
    
    async def get_supported_regions(self) -> List[str]:
        """
        Get list of supported regions for this provider.
        
        Returns:
            List[str]: List of supported region identifiers
        """
        # Default implementation - subclasses should override
        return [self.default_region]
    
    async def get_supported_instance_types(self, region: str) -> List[str]:
        """
        Get list of supported instance types for the region.
        
        Args:
            region: Region identifier
            
        Returns:
            List[str]: List of supported instance types
        """
        # Default implementation - subclasses should override
        return []
    
    def _normalize_region(self, region: str) -> str:
        """
        Normalize region identifier to provider-specific format.
        
        Args:
            region: Region identifier
            
        Returns:
            str: Normalized region identifier
        """
        # Default implementation returns as-is
        # Subclasses can override for provider-specific normalization
        return region
    
    def _handle_rate_limiting(self):
        """Handle rate limiting for API calls."""
        # Default implementation - subclasses can override
        pass
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the pricing API.
        
        Returns:
            bool: True if API is healthy, False otherwise
        """
        try:
            # Try to get pricing for default region with minimal data
            query = PricingQuery(
                provider=self.provider_name,
                service_name="compute",
                region=self.default_region,
                filters={"limit": 1}
            )
            
            response = await self.get_comprehensive_pricing(query)
            return response.success
            
        except Exception as e:
            logger.error(f"Health check failed for {self.provider_name}: {e}")
            return False


class PricingAPIException(Exception):
    """Exception raised when pricing API calls fail."""
    
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} pricing API error: {message}")


class RateLimitException(PricingAPIException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        message = f"Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(message, provider, 429)