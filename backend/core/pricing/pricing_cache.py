"""
Pricing Data Cache

Implements caching for pricing data to improve performance and reduce API calls.
Uses Redis for distributed caching with appropriate TTL values.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
from decimal import Decimal

from .pricing_models import PricingData, PricingResponse

logger = logging.getLogger(__name__)


class PricingDataCache:
    """
    Pricing data cache implementation with Redis backend.
    
    Provides intelligent caching for pricing data with configurable TTL values
    and cache invalidation strategies.
    """
    
    # Cache TTL values (in seconds)
    DEFAULT_TTL = 3600  # 1 hour
    CACHE_TTLS = {
        'compute': 3600,    # 1 hour - compute pricing changes infrequently
        'storage': 7200,    # 2 hours - storage pricing is very stable
        'network': 1800,    # 30 minutes - network pricing can vary
        'database': 3600,   # 1 hour - database pricing is stable
        'spot': 300,        # 5 minutes - spot pricing changes frequently
        'reserved': 86400   # 24 hours - reserved pricing changes rarely
    }
    
    # Cache key prefixes
    KEY_PREFIXES = {
        'pricing': 'pricing',
        'comparison': 'comparison',
        'tco': 'tco',
        'migration': 'migration'
    }
    
    def __init__(self, redis_client=None, key_prefix: str = "finops"):
        """
        Initialize pricing data cache.
        
        Args:
            redis_client: Redis client instance (optional for mock)
            key_prefix: Prefix for all cache keys
        """
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.mock_cache = {}  # In-memory cache for testing/demo
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        
        logger.info(f"Initialized pricing cache with prefix: {key_prefix}")
    
    def _generate_cache_key(self, key_type: str, provider: str, **kwargs) -> str:
        """
        Generate cache key for pricing data.
        
        Args:
            key_type: Type of cache key (pricing, comparison, etc.)
            provider: Cloud provider name
            **kwargs: Additional key parameters
            
        Returns:
            str: Generated cache key
        """
        # Create a deterministic key from parameters
        key_parts = [self.key_prefix, self.KEY_PREFIXES.get(key_type, key_type), provider]
        
        # Add sorted kwargs to ensure consistent key generation
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        
        cache_key = ":".join(key_parts)
        
        # Hash long keys to avoid Redis key length limits
        if len(cache_key) > 250:
            key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_key = f"{self.key_prefix}:hash:{key_hash}"
        
        return cache_key
    
    async def get_cached_pricing(
        self, 
        provider: str, 
        service_name: str, 
        region: str,
        **filters
    ) -> Optional[PricingData]:
        """
        Get cached pricing data.
        
        Args:
            provider: Cloud provider name
            service_name: Service name
            region: Region identifier
            **filters: Additional filters
            
        Returns:
            Optional[PricingData]: Cached pricing data if available
        """
        try:
            cache_key = self._generate_cache_key(
                'pricing', provider,
                service=service_name,
                region=region,
                **filters
            )
            
            cached_data = await self._get_from_cache(cache_key)
            
            if cached_data:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                
                # Deserialize pricing data
                pricing_data = self._deserialize_pricing_data(cached_data)
                return pricing_data
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached pricing: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def cache_pricing_data(
        self, 
        provider: str, 
        service_name: str, 
        region: str,
        pricing_data: PricingData,
        ttl: Optional[int] = None,
        **filters
    ):
        """
        Cache pricing data.
        
        Args:
            provider: Cloud provider name
            service_name: Service name
            region: Region identifier
            pricing_data: Pricing data to cache
            ttl: Time to live in seconds (optional)
            **filters: Additional filters
        """
        try:
            cache_key = self._generate_cache_key(
                'pricing', provider,
                service=service_name,
                region=region,
                **filters
            )
            
            # Determine TTL based on service type
            if ttl is None:
                ttl = self._get_ttl_for_service(service_name)
            
            # Serialize pricing data
            serialized_data = self._serialize_pricing_data(pricing_data)
            
            # Cache the data
            await self._set_in_cache(cache_key, serialized_data, ttl)
            
            self.cache_stats['sets'] += 1
            logger.debug(f"Cached pricing data with key: {cache_key}, TTL: {ttl}s")
            
        except Exception as e:
            logger.error(f"Failed to cache pricing data: {e}")
    
    async def get_cached_comparison(self, workload_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached cost comparison result.
        
        Args:
            workload_hash: Hash of workload specification
            
        Returns:
            Optional[Dict]: Cached comparison data if available
        """
        try:
            cache_key = self._generate_cache_key('comparison', 'multi', workload=workload_hash)
            
            cached_data = await self._get_from_cache(cache_key)
            
            if cached_data:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit for comparison: {workload_hash}")
                return cached_data
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"Cache miss for comparison: {workload_hash}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached comparison: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def cache_comparison_result(
        self, 
        workload_hash: str, 
        comparison_data: Dict[str, Any],
        ttl: int = 1800  # 30 minutes default
    ):
        """
        Cache cost comparison result.
        
        Args:
            workload_hash: Hash of workload specification
            comparison_data: Comparison result to cache
            ttl: Time to live in seconds
        """
        try:
            cache_key = self._generate_cache_key('comparison', 'multi', workload=workload_hash)
            
            await self._set_in_cache(cache_key, comparison_data, ttl)
            
            self.cache_stats['sets'] += 1
            logger.debug(f"Cached comparison result: {workload_hash}")
            
        except Exception as e:
            logger.error(f"Failed to cache comparison result: {e}")
    
    async def get_cached_tco(self, workload_hash: str, years: int) -> Optional[Dict[str, Any]]:
        """
        Get cached TCO analysis result.
        
        Args:
            workload_hash: Hash of workload specification
            years: Time horizon for TCO analysis
            
        Returns:
            Optional[Dict]: Cached TCO data if available
        """
        try:
            cache_key = self._generate_cache_key('tco', 'multi', workload=workload_hash, years=years)
            
            cached_data = await self._get_from_cache(cache_key)
            
            if cached_data:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit for TCO: {workload_hash}")
                return cached_data
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"Cache miss for TCO: {workload_hash}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached TCO: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def cache_tco_result(
        self, 
        workload_hash: str, 
        years: int,
        tco_data: Dict[str, Any],
        ttl: int = 7200  # 2 hours default
    ):
        """
        Cache TCO analysis result.
        
        Args:
            workload_hash: Hash of workload specification
            years: Time horizon for TCO analysis
            tco_data: TCO result to cache
            ttl: Time to live in seconds
        """
        try:
            cache_key = self._generate_cache_key('tco', 'multi', workload=workload_hash, years=years)
            
            await self._set_in_cache(cache_key, tco_data, ttl)
            
            self.cache_stats['sets'] += 1
            logger.debug(f"Cached TCO result: {workload_hash}")
            
        except Exception as e:
            logger.error(f"Failed to cache TCO result: {e}")
    
    async def invalidate_pricing_cache(
        self, 
        provider: str, 
        service_name: Optional[str] = None,
        region: Optional[str] = None
    ):
        """
        Invalidate pricing cache for a provider/service/region.
        
        Args:
            provider: Cloud provider name
            service_name: Service name (optional, invalidates all if None)
            region: Region identifier (optional, invalidates all if None)
        """
        try:
            # Generate pattern for cache keys to invalidate
            if service_name and region:
                # Invalidate specific service in specific region
                cache_key = self._generate_cache_key('pricing', provider, service=service_name, region=region)
                await self._delete_from_cache(cache_key)
                self.cache_stats['deletes'] += 1
            else:
                # Invalidate all pricing data for provider (pattern-based deletion)
                pattern = f"{self.key_prefix}:{self.KEY_PREFIXES['pricing']}:{provider}:*"
                deleted_count = await self._delete_pattern_from_cache(pattern)
                self.cache_stats['deletes'] += deleted_count
            
            logger.info(f"Invalidated pricing cache for {provider}/{service_name}/{region}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate pricing cache: {e}")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dict: Cache statistics including hit rate, miss rate, etc.
        """
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        miss_rate = (self.cache_stats['misses'] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'total_requests': total_requests,
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'miss_rate_percent': round(miss_rate, 2),
            'cache_sets': self.cache_stats['sets'],
            'cache_deletes': self.cache_stats['deletes'],
            'backend': 'redis' if self.redis_client else 'memory'
        }
        
        # Add Redis-specific stats if available
        if self.redis_client:
            try:
                redis_info = await self._get_redis_info()
                stats.update(redis_info)
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
        
        return stats
    
    def _get_ttl_for_service(self, service_name: str) -> int:
        """Get appropriate TTL for service type."""
        service_lower = service_name.lower()
        
        for service_type, ttl in self.CACHE_TTLS.items():
            if service_type in service_lower:
                return ttl
        
        return self.DEFAULT_TTL
    
    def _serialize_pricing_data(self, pricing_data: PricingData) -> Dict[str, Any]:
        """Serialize pricing data for caching."""
        # Convert PricingData to dictionary with JSON-serializable values
        data = {
            'provider': pricing_data.provider,
            'service_name': pricing_data.service_name,
            'service_category': pricing_data.service_category.value,
            'region': pricing_data.region,
            'pricing_date': pricing_data.pricing_date.isoformat(),
            'compute_pricing': [self._serialize_compute_pricing(cp) for cp in pricing_data.compute_pricing],
            'storage_pricing': [self._serialize_storage_pricing(sp) for sp in pricing_data.storage_pricing],
            'network_pricing': [self._serialize_network_pricing(np) for np in pricing_data.network_pricing],
            'database_pricing': [self._serialize_database_pricing(dp) for dp in pricing_data.database_pricing],
            'raw_data': pricing_data.raw_data
        }
        return data
    
    def _deserialize_pricing_data(self, data: Dict[str, Any]) -> PricingData:
        """Deserialize pricing data from cache."""
        from .pricing_models import ServiceCategory
        
        pricing_data = PricingData(
            provider=data['provider'],
            service_name=data['service_name'],
            service_category=ServiceCategory(data['service_category']),
            region=data['region'],
            pricing_date=datetime.fromisoformat(data['pricing_date']),
            compute_pricing=[self._deserialize_compute_pricing(cp) for cp in data['compute_pricing']],
            storage_pricing=[self._deserialize_storage_pricing(sp) for sp in data['storage_pricing']],
            network_pricing=[self._deserialize_network_pricing(np) for np in data['network_pricing']],
            database_pricing=[self._deserialize_database_pricing(dp) for dp in data['database_pricing']],
            raw_data=data['raw_data']
        )
        return pricing_data
    
    def _serialize_compute_pricing(self, cp) -> Dict[str, Any]:
        """Serialize compute pricing."""
        return {
            'instance_type': cp.instance_type,
            'vcpus': cp.vcpus,
            'memory_gb': cp.memory_gb,
            'price_per_hour': str(cp.price_per_hour),
            'price_per_month': str(cp.price_per_month),
            'operating_system': cp.operating_system,
            'region': cp.region,
            'currency': cp.currency,
            'spot_price_per_hour': str(cp.spot_price_per_hour) if cp.spot_price_per_hour else None,
            'reserved_price_per_hour': str(cp.reserved_price_per_hour) if cp.reserved_price_per_hour else None,
            'architecture': cp.architecture,
            'network_performance': cp.network_performance,
            'storage_type': cp.storage_type,
            'additional_specs': cp.additional_specs
        }
    
    def _deserialize_compute_pricing(self, data: Dict[str, Any]):
        """Deserialize compute pricing."""
        from .pricing_models import ComputePricing
        
        return ComputePricing(
            instance_type=data['instance_type'],
            vcpus=data['vcpus'],
            memory_gb=data['memory_gb'],
            price_per_hour=Decimal(data['price_per_hour']),
            price_per_month=Decimal(data['price_per_month']),
            operating_system=data['operating_system'],
            region=data['region'],
            currency=data['currency'],
            spot_price_per_hour=Decimal(data['spot_price_per_hour']) if data['spot_price_per_hour'] else None,
            reserved_price_per_hour=Decimal(data['reserved_price_per_hour']) if data['reserved_price_per_hour'] else None,
            architecture=data['architecture'],
            network_performance=data['network_performance'],
            storage_type=data['storage_type'],
            additional_specs=data['additional_specs']
        )
    
    def _serialize_storage_pricing(self, sp) -> Dict[str, Any]:
        """Serialize storage pricing."""
        return {
            'storage_type': sp.storage_type,
            'price_per_gb_month': str(sp.price_per_gb_month),
            'region': sp.region,
            'currency': sp.currency,
            'storage_class': sp.storage_class,
            'iops_price': str(sp.iops_price) if sp.iops_price else None,
            'throughput_price': str(sp.throughput_price) if sp.throughput_price else None,
            'request_price': str(sp.request_price) if sp.request_price else None,
            'retrieval_price': str(sp.retrieval_price) if sp.retrieval_price else None,
            'minimum_storage_duration': sp.minimum_storage_duration,
            'additional_specs': sp.additional_specs
        }
    
    def _deserialize_storage_pricing(self, data: Dict[str, Any]):
        """Deserialize storage pricing."""
        from .pricing_models import StoragePricing
        
        return StoragePricing(
            storage_type=data['storage_type'],
            price_per_gb_month=Decimal(data['price_per_gb_month']),
            region=data['region'],
            currency=data['currency'],
            storage_class=data['storage_class'],
            iops_price=Decimal(data['iops_price']) if data['iops_price'] else None,
            throughput_price=Decimal(data['throughput_price']) if data['throughput_price'] else None,
            request_price=Decimal(data['request_price']) if data['request_price'] else None,
            retrieval_price=Decimal(data['retrieval_price']) if data['retrieval_price'] else None,
            minimum_storage_duration=data['minimum_storage_duration'],
            additional_specs=data['additional_specs']
        )
    
    def _serialize_network_pricing(self, np) -> Dict[str, Any]:
        """Serialize network pricing."""
        return {
            'service_type': np.service_type,
            'price_per_gb': str(np.price_per_gb) if np.price_per_gb else None,
            'price_per_hour': str(np.price_per_hour) if np.price_per_hour else None,
            'price_per_request': str(np.price_per_request) if np.price_per_request else None,
            'region': np.region,
            'currency': np.currency,
            'transfer_type': np.transfer_type,
            'bandwidth_tier': np.bandwidth_tier,
            'additional_specs': np.additional_specs
        }
    
    def _deserialize_network_pricing(self, data: Dict[str, Any]):
        """Deserialize network pricing."""
        from .pricing_models import NetworkPricing
        
        return NetworkPricing(
            service_type=data['service_type'],
            price_per_gb=Decimal(data['price_per_gb']) if data['price_per_gb'] else None,
            price_per_hour=Decimal(data['price_per_hour']) if data['price_per_hour'] else None,
            price_per_request=Decimal(data['price_per_request']) if data['price_per_request'] else None,
            region=data['region'],
            currency=data['currency'],
            transfer_type=data['transfer_type'],
            bandwidth_tier=data['bandwidth_tier'],
            additional_specs=data['additional_specs']
        )
    
    def _serialize_database_pricing(self, dp) -> Dict[str, Any]:
        """Serialize database pricing."""
        return {
            'database_type': dp.database_type,
            'instance_class': dp.instance_class,
            'price_per_hour': str(dp.price_per_hour),
            'storage_price_per_gb_month': str(dp.storage_price_per_gb_month),
            'region': dp.region,
            'currency': dp.currency,
            'engine_version': dp.engine_version,
            'multi_az': dp.multi_az,
            'backup_storage_price': str(dp.backup_storage_price) if dp.backup_storage_price else None,
            'iops_price': str(dp.iops_price) if dp.iops_price else None,
            'additional_specs': dp.additional_specs
        }
    
    def _deserialize_database_pricing(self, data: Dict[str, Any]):
        """Deserialize database pricing."""
        from .pricing_models import DatabasePricing
        
        return DatabasePricing(
            database_type=data['database_type'],
            instance_class=data['instance_class'],
            price_per_hour=Decimal(data['price_per_hour']),
            storage_price_per_gb_month=Decimal(data['storage_price_per_gb_month']),
            region=data['region'],
            currency=data['currency'],
            engine_version=data['engine_version'],
            multi_az=data['multi_az'],
            backup_storage_price=Decimal(data['backup_storage_price']) if data['backup_storage_price'] else None,
            iops_price=Decimal(data['iops_price']) if data['iops_price'] else None,
            additional_specs=data['additional_specs']
        )
    
    # Cache backend methods (Redis or in-memory)
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache backend."""
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data)
                return None
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                return None
        else:
            # Use in-memory cache for testing/demo
            cache_entry = self.mock_cache.get(key)
            if cache_entry:
                # Check if expired
                if datetime.utcnow() < cache_entry['expires_at']:
                    return cache_entry['data']
                else:
                    # Remove expired entry
                    del self.mock_cache[key]
            return None
    
    async def _set_in_cache(self, key: str, data: Dict[str, Any], ttl: int):
        """Set data in cache backend."""
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(data, default=str))
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        else:
            # Use in-memory cache for testing/demo
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            self.mock_cache[key] = {
                'data': data,
                'expires_at': expires_at
            }
    
    async def _delete_from_cache(self, key: str):
        """Delete data from cache backend."""
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        else:
            # Use in-memory cache for testing/demo
            self.mock_cache.pop(key, None)
    
    async def _delete_pattern_from_cache(self, pattern: str) -> int:
        """Delete keys matching pattern from cache backend."""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    return len(keys)
                return 0
            except Exception as e:
                logger.error(f"Redis pattern delete error: {e}")
                return 0
        else:
            # Use in-memory cache for testing/demo
            import fnmatch
            keys_to_delete = [key for key in self.mock_cache.keys() if fnmatch.fnmatch(key, pattern)]
            for key in keys_to_delete:
                del self.mock_cache[key]
            return len(keys_to_delete)
    
    async def _get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        if self.redis_client:
            try:
                info = await self.redis_client.info('memory')
                return {
                    'redis_memory_used': info.get('used_memory_human', 'unknown'),
                    'redis_memory_peak': info.get('used_memory_peak_human', 'unknown'),
                    'redis_connected_clients': info.get('connected_clients', 0)
                }
            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
        return {}
    
    async def close(self):
        """Close cache connections."""
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        # Clear in-memory cache
        self.mock_cache.clear()