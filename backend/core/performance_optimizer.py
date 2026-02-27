"""
Performance Optimizer for Multi-Cloud Cost Engine

Provides intelligent caching strategies, parallel processing for provider API calls,
and performance optimization techniques for the multi-cloud cost comparison engine.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class RequestPattern(Enum):
    """Types of request patterns for cache optimization"""
    FREQUENT_IDENTICAL = "frequent_identical"
    SIMILAR_WORKLOADS = "similar_workloads"
    PROVIDER_SPECIFIC = "provider_specific"
    TIME_SENSITIVE = "time_sensitive"
    BATCH_PROCESSING = "batch_processing"


@dataclass
class CacheStrategy:
    """Cache strategy configuration"""
    ttl_seconds: int
    max_entries: int
    eviction_policy: str = "lru"
    compression_enabled: bool = False
    distributed: bool = False
    cache_layers: List[str] = field(default_factory=lambda: ["memory", "redis"])
    invalidation_triggers: List[str] = field(default_factory=list)


@dataclass
class PricingRequest:
    """Pricing request structure"""
    provider: str
    service: str
    region: str
    instance_type: Optional[str] = None
    pricing_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PricingResponse:
    """Pricing response structure"""
    request: PricingRequest
    price_per_unit: Decimal
    currency: str
    effective_date: datetime
    response_time_ms: float
    cached: bool = False
    cache_age_seconds: Optional[float] = None


@dataclass
class WorkloadSpec:
    """Simplified workload specification for performance optimization"""
    compute_cores: int
    memory_gb: int
    storage_gb: int
    network_gb_monthly: int
    regions: List[str]
    providers: List[str]
    complexity_score: float = 1.0


class PerformanceOptimizer:
    """
    Performance optimizer for multi-cloud cost comparison operations.
    
    Provides intelligent caching, parallel processing, and optimization
    strategies to improve system performance and scalability.
    """
    
    def __init__(self, cache_manager=None, metrics_collector=None):
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        self.request_patterns = {}
        self.optimization_history = []
        
        # Performance thresholds
        self.max_parallel_requests = 10
        self.request_timeout_seconds = 30
        self.cache_warming_threshold = 0.7  # Warm cache when hit rate drops below 70%
        
        # Provider-specific optimization settings
        self.provider_configs = {
            'aws': {
                'max_concurrent_requests': 5,
                'rate_limit_per_second': 10,
                'preferred_regions': ['us-east-1', 'us-west-2'],
                'cache_ttl_seconds': 3600
            },
            'gcp': {
                'max_concurrent_requests': 4,
                'rate_limit_per_second': 8,
                'preferred_regions': ['us-central1', 'europe-west1'],
                'cache_ttl_seconds': 3600
            },
            'azure': {
                'max_concurrent_requests': 4,
                'rate_limit_per_second': 8,
                'preferred_regions': ['East US', 'West Europe'],
                'cache_ttl_seconds': 3600
            }
        }
    
    async def parallel_provider_requests(self, workload: WorkloadSpec) -> Dict[str, Any]:
        """
        Execute pricing requests to multiple providers in parallel.
        
        Args:
            workload: Workload specification for pricing requests
            
        Returns:
            Dictionary containing pricing data from all providers
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Starting parallel provider requests",
                providers=workload.providers,
                regions=len(workload.regions)
            )
            
            # Create pricing requests for each provider
            provider_tasks = []
            for provider in workload.providers:
                task = self._fetch_provider_pricing(provider, workload)
                provider_tasks.append((provider, task))
            
            # Execute requests in parallel with timeout
            results = {}
            completed_tasks = await asyncio.gather(
                *[task for _, task in provider_tasks],
                return_exceptions=True
            )
            
            # Process results
            for i, (provider, _) in enumerate(provider_tasks):
                result = completed_tasks[i]
                if isinstance(result, Exception):
                    logger.error(
                        "Provider request failed",
                        provider=provider,
                        error=str(result)
                    )
                    results[provider] = {
                        'error': str(result),
                        'pricing_data': None,
                        'response_time_ms': None
                    }
                else:
                    results[provider] = result
            
            total_time = (time.time() - start_time) * 1000
            
            # Track performance metrics
            if self.metrics_collector:
                self.metrics_collector.track_parallel_request_performance(
                    provider_count=len(workload.providers),
                    total_time_ms=total_time,
                    success_count=len([r for r in results.values() if 'error' not in r])
                )
            
            logger.info(
                "Parallel provider requests completed",
                total_time_ms=total_time,
                successful_providers=len([r for r in results.values() if 'error' not in r])
            )
            
            return {
                'results': results,
                'total_time_ms': total_time,
                'parallel_efficiency': self._calculate_parallel_efficiency(results, total_time)
            }
            
        except Exception as e:
            logger.error("Parallel provider requests failed", error=str(e))
            raise
    
    async def _fetch_provider_pricing(self, provider: str, workload: WorkloadSpec) -> Dict[str, Any]:
        """Fetch pricing data from a specific provider"""
        start_time = time.time()
        
        try:
            # Get provider configuration
            config = self.provider_configs.get(provider, {})
            max_concurrent = config.get('max_concurrent_requests', 3)
            
            # Create pricing requests for each region
            region_tasks = []
            for region in workload.regions:
                # Limit concurrent requests per provider
                if len(region_tasks) >= max_concurrent:
                    break
                
                task = self._fetch_region_pricing(provider, region, workload)
                region_tasks.append((region, task))
            
            # Execute region requests with semaphore for rate limiting
            semaphore = asyncio.Semaphore(max_concurrent)
            region_results = {}
            
            async def fetch_with_semaphore(region, task):
                async with semaphore:
                    return await task
            
            region_responses = await asyncio.gather(
                *[fetch_with_semaphore(region, task) for region, task in region_tasks],
                return_exceptions=True
            )
            
            # Process region results
            for i, (region, _) in enumerate(region_tasks):
                response = region_responses[i]
                if isinstance(response, Exception):
                    logger.warning(
                        "Region pricing request failed",
                        provider=provider,
                        region=region,
                        error=str(response)
                    )
                    region_results[region] = None
                else:
                    region_results[region] = response
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return {
                'provider': provider,
                'pricing_data': region_results,
                'response_time_ms': response_time_ms,
                'regions_processed': len([r for r in region_results.values() if r is not None])
            }
            
        except Exception as e:
            logger.error(
                "Provider pricing fetch failed",
                provider=provider,
                error=str(e)
            )
            raise
    
    async def _fetch_region_pricing(self, provider: str, region: str, workload: WorkloadSpec) -> Dict[str, Any]:
        """Fetch pricing data for a specific provider and region"""
        
        # Generate cache key
        cache_key = self._generate_pricing_cache_key(provider, region, workload)
        
        # Try to get from cache first
        if self.cache_manager:
            cached_result = await self.cache_manager.get_cached_pricing(cache_key)
            if cached_result:
                return {
                    'region': region,
                    'pricing': cached_result,
                    'cached': True,
                    'cache_age_seconds': cached_result.get('cache_age', 0)
                }
        
        # Mock pricing calculation (in production, this would call actual provider APIs)
        await asyncio.sleep(0.1)  # Simulate API call delay
        
        # Calculate mock pricing based on workload
        base_compute_cost = Decimal(str(workload.compute_cores * 0.096))  # $0.096 per core-hour
        base_memory_cost = Decimal(str(workload.memory_gb * 0.012))       # $0.012 per GB-hour
        base_storage_cost = Decimal(str(workload.storage_gb * 0.023))     # $0.023 per GB-month
        base_network_cost = Decimal(str(workload.network_gb_monthly * 0.09))  # $0.09 per GB
        
        # Apply provider-specific multipliers
        provider_multipliers = {
            'aws': Decimal('1.0'),
            'gcp': Decimal('0.95'),
            'azure': Decimal('1.05')
        }
        
        multiplier = provider_multipliers.get(provider, Decimal('1.0'))
        total_monthly_cost = (base_compute_cost + base_memory_cost + base_storage_cost + base_network_cost) * multiplier
        
        pricing_result = {
            'region': region,
            'pricing': {
                'compute_cost': base_compute_cost * multiplier,
                'memory_cost': base_memory_cost * multiplier,
                'storage_cost': base_storage_cost * multiplier,
                'network_cost': base_network_cost * multiplier,
                'total_monthly_cost': total_monthly_cost,
                'currency': 'USD',
                'effective_date': datetime.utcnow().isoformat()
            },
            'cached': False,
            'cache_age_seconds': 0
        }
        
        # Cache the result
        if self.cache_manager:
            config = self.provider_configs.get(provider, {})
            ttl = config.get('cache_ttl_seconds', 3600)
            await self.cache_manager.cache_pricing_data(cache_key, pricing_result, ttl)
        
        return pricing_result
    
    def optimize_cache_strategy(self, request_pattern: RequestPattern) -> CacheStrategy:
        """
        Optimize cache strategy based on request patterns.
        
        Args:
            request_pattern: Detected request pattern
            
        Returns:
            Optimized cache strategy
        """
        logger.info("Optimizing cache strategy", pattern=request_pattern.value)
        
        # Base cache strategy
        base_strategy = CacheStrategy(
            ttl_seconds=3600,
            max_entries=10000,
            eviction_policy="lru",
            compression_enabled=False,
            distributed=False
        )
        
        # Optimize based on pattern
        if request_pattern == RequestPattern.FREQUENT_IDENTICAL:
            # High cache hit rate expected, longer TTL
            return CacheStrategy(
                ttl_seconds=7200,  # 2 hours
                max_entries=5000,
                eviction_policy="lru",
                compression_enabled=True,
                distributed=True,
                cache_layers=["memory", "redis"],
                invalidation_triggers=["pricing_update"]
            )
        
        elif request_pattern == RequestPattern.SIMILAR_WORKLOADS:
            # Medium cache hit rate, moderate TTL
            return CacheStrategy(
                ttl_seconds=3600,  # 1 hour
                max_entries=15000,
                eviction_policy="lfu",  # Least Frequently Used
                compression_enabled=True,
                distributed=True,
                cache_layers=["memory", "redis"],
                invalidation_triggers=["pricing_update", "workload_pattern_change"]
            )
        
        elif request_pattern == RequestPattern.PROVIDER_SPECIFIC:
            # Provider-focused caching
            return CacheStrategy(
                ttl_seconds=1800,  # 30 minutes
                max_entries=8000,
                eviction_policy="lru",
                compression_enabled=False,
                distributed=True,
                cache_layers=["redis"],  # Skip memory cache for provider-specific
                invalidation_triggers=["provider_pricing_update"]
            )
        
        elif request_pattern == RequestPattern.TIME_SENSITIVE:
            # Short TTL for time-sensitive requests
            return CacheStrategy(
                ttl_seconds=900,  # 15 minutes
                max_entries=20000,
                eviction_policy="ttl",  # Time-based eviction
                compression_enabled=False,
                distributed=False,
                cache_layers=["memory"],
                invalidation_triggers=["time_threshold"]
            )
        
        elif request_pattern == RequestPattern.BATCH_PROCESSING:
            # Large cache for batch operations
            return CacheStrategy(
                ttl_seconds=14400,  # 4 hours
                max_entries=50000,
                eviction_policy="lru",
                compression_enabled=True,
                distributed=True,
                cache_layers=["memory", "redis", "disk"],
                invalidation_triggers=["batch_completion", "pricing_update"]
            )
        
        return base_strategy
    
    async def batch_pricing_requests(self, requests: List[PricingRequest]) -> List[PricingResponse]:
        """
        Process multiple pricing requests in optimized batches.
        
        Args:
            requests: List of pricing requests to process
            
        Returns:
            List of pricing responses
        """
        start_time = time.time()
        
        logger.info("Starting batch pricing requests", request_count=len(requests))
        
        try:
            # Group requests by provider for optimal batching
            provider_groups = {}
            for request in requests:
                provider = request.provider
                if provider not in provider_groups:
                    provider_groups[provider] = []
                provider_groups[provider].append(request)
            
            # Process each provider group in parallel
            provider_tasks = []
            for provider, provider_requests in provider_groups.items():
                task = self._process_provider_batch(provider, provider_requests)
                provider_tasks.append(task)
            
            # Execute provider batches in parallel
            provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)
            
            # Flatten results
            all_responses = []
            for result in provider_results:
                if isinstance(result, Exception):
                    logger.error("Provider batch failed", error=str(result))
                    continue
                all_responses.extend(result)
            
            total_time = (time.time() - start_time) * 1000
            
            # Track batch performance
            if self.metrics_collector:
                self.metrics_collector.track_batch_performance(
                    request_count=len(requests),
                    response_count=len(all_responses),
                    total_time_ms=total_time
                )
            
            logger.info(
                "Batch pricing requests completed",
                total_requests=len(requests),
                successful_responses=len(all_responses),
                total_time_ms=total_time
            )
            
            return all_responses
            
        except Exception as e:
            logger.error("Batch pricing requests failed", error=str(e))
            raise
    
    async def _process_provider_batch(self, provider: str, requests: List[PricingRequest]) -> List[PricingResponse]:
        """Process a batch of requests for a specific provider"""
        
        config = self.provider_configs.get(provider, {})
        max_concurrent = config.get('max_concurrent_requests', 3)
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(request: PricingRequest) -> PricingResponse:
            async with semaphore:
                return await self._process_single_pricing_request(request)
        
        # Process requests with rate limiting
        tasks = [process_single_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid responses
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.warning(
                    "Single pricing request failed",
                    provider=provider,
                    request=requests[i],
                    error=str(response)
                )
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    async def _process_single_pricing_request(self, request: PricingRequest) -> PricingResponse:
        """Process a single pricing request"""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_request_cache_key(request)
        
        # Check cache first
        cached_response = None
        if self.cache_manager:
            cached_response = await self.cache_manager.get_cached_pricing(cache_key)
        
        if cached_response:
            response_time = (time.time() - start_time) * 1000
            return PricingResponse(
                request=request,
                price_per_unit=cached_response['price_per_unit'],
                currency=cached_response['currency'],
                effective_date=cached_response['effective_date'],
                response_time_ms=response_time,
                cached=True,
                cache_age_seconds=cached_response.get('cache_age', 0)
            )
        
        # Mock API call (in production, this would call actual provider API)
        await asyncio.sleep(0.05)  # Simulate API delay
        
        # Calculate mock pricing
        base_price = Decimal('0.096')  # Base price per unit
        provider_multipliers = {'aws': 1.0, 'gcp': 0.95, 'azure': 1.05}
        multiplier = provider_multipliers.get(request.provider, 1.0)
        
        price_per_unit = base_price * Decimal(str(multiplier))
        response_time = (time.time() - start_time) * 1000
        
        response = PricingResponse(
            request=request,
            price_per_unit=price_per_unit,
            currency='USD',
            effective_date=datetime.utcnow(),
            response_time_ms=response_time,
            cached=False
        )
        
        # Cache the response
        if self.cache_manager:
            cache_data = {
                'price_per_unit': price_per_unit,
                'currency': 'USD',
                'effective_date': datetime.utcnow(),
                'cache_age': 0
            }
            await self.cache_manager.cache_pricing_data(cache_key, cache_data, 3600)
        
        return response
    
    def _generate_pricing_cache_key(self, provider: str, region: str, workload: WorkloadSpec) -> str:
        """Generate cache key for pricing data"""
        key_data = f"{provider}:{region}:{workload.compute_cores}:{workload.memory_gb}:{workload.storage_gb}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_request_cache_key(self, request: PricingRequest) -> str:
        """Generate cache key for pricing request"""
        key_data = f"{request.provider}:{request.service}:{request.region}:{request.instance_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_parallel_efficiency(self, results: Dict[str, Any], total_time_ms: float) -> float:
        """Calculate parallel processing efficiency"""
        successful_results = [r for r in results.values() if 'error' not in r]
        
        if not successful_results:
            return 0.0
        
        # Calculate average sequential time
        avg_sequential_time = sum(
            r.get('response_time_ms', 0) for r in successful_results
        ) / len(successful_results)
        
        # Calculate theoretical sequential total
        theoretical_sequential_total = avg_sequential_time * len(successful_results)
        
        # Calculate efficiency (higher is better)
        if total_time_ms > 0:
            efficiency = min(theoretical_sequential_total / total_time_ms, 10.0)  # Cap at 10x
            return efficiency
        
        return 1.0
    
    def analyze_request_patterns(self, request_history: List[Dict[str, Any]]) -> RequestPattern:
        """
        Analyze request history to determine optimal caching pattern.
        
        Args:
            request_history: List of historical requests
            
        Returns:
            Detected request pattern
        """
        if not request_history:
            return RequestPattern.FREQUENT_IDENTICAL
        
        # Analyze patterns
        unique_requests = len(set(str(req) for req in request_history))
        total_requests = len(request_history)
        
        # Calculate similarity ratio
        similarity_ratio = 1 - (unique_requests / total_requests)
        
        # Analyze provider distribution
        provider_counts = {}
        for req in request_history:
            provider = req.get('provider', 'unknown')
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        dominant_provider_ratio = max(provider_counts.values()) / total_requests if provider_counts else 0
        
        # Analyze time distribution
        recent_requests = [
            req for req in request_history 
            if req.get('timestamp', 0) > time.time() - 3600  # Last hour
        ]
        time_concentration = len(recent_requests) / total_requests if total_requests > 0 else 0
        
        # Determine pattern
        if similarity_ratio > 0.8:
            return RequestPattern.FREQUENT_IDENTICAL
        elif dominant_provider_ratio > 0.7:
            return RequestPattern.PROVIDER_SPECIFIC
        elif time_concentration > 0.8:
            return RequestPattern.TIME_SENSITIVE
        elif total_requests > 100:
            return RequestPattern.BATCH_PROCESSING
        else:
            return RequestPattern.SIMILAR_WORKLOADS