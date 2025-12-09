"""
AWS Pricing API Service

Fetches real-time pricing data from AWS Pricing API instead of using hardcoded values.
Supports EC2, RDS, and other services with caching for performance.
"""

import boto3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)


class ServiceCode(Enum):
    """AWS Service codes for pricing API"""
    EC2 = "AmazonEC2"
    RDS = "AmazonRDS"
    S3 = "AmazonS3"
    LAMBDA = "AWSLambda"
    DYNAMODB = "AmazonDynamoDB"


class PricingCache:
    """Cache for pricing data to reduce API calls"""
    
    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_hours = ttl_hours
        self.last_updated: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached pricing data"""
        if key in self.cache:
            if key in self.last_updated:
                age = datetime.now() - self.last_updated[key]
                if age < timedelta(hours=self.ttl_hours):
                    return self.cache[key]
                else:
                    # Cache expired
                    del self.cache[key]
                    del self.last_updated[key]
        return None
    
    def set(self, key: str, value: Dict[str, Any]):
        """Set cached pricing data"""
        self.cache[key] = value
        self.last_updated[key] = datetime.now()
    
    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
        self.last_updated.clear()


class AWSPricingService:
    """
    Service for fetching real-time AWS pricing data.
    Uses AWS Pricing API with intelligent caching.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.pricing_client = None
        self.cache = PricingCache(ttl_hours=24)
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def _get_pricing_client(self):
        """Get AWS Pricing client (lazy initialization)"""
        if not self.pricing_client:
            # Pricing API is only available in us-east-1 and ap-south-1
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        return self.pricing_client
    
    async def get_ec2_on_demand_pricing(self, 
                                       instance_type: str, 
                                       region: str,
                                       operating_system: str = 'Linux') -> Optional[Decimal]:
        """
        Get EC2 on-demand hourly pricing for specific instance type and region.
        
        Args:
            instance_type: EC2 instance type (e.g., 'm5.large')
            region: AWS region (e.g., 'us-east-1')
            operating_system: OS type (Linux, Windows, RHEL, SUSE)
            
        Returns:
            Hourly price as Decimal or None if not found
        """
        cache_key = f"ec2_ondemand_{instance_type}_{region}_{operating_system}"
        
        # Check cache first
        cached_price = self.cache.get(cache_key)
        if cached_price:
            logger.debug("Using cached EC2 pricing", 
                        instance_type=instance_type, 
                        region=region)
            return Decimal(str(cached_price['price']))
        
        try:
            def _fetch_price():
                client = self._get_pricing_client()
                
                # Convert region code to region name
                region_name = self._get_region_name(region)
                
                # Build filters for pricing query
                filters = [
                    {'Type': 'TERM_MATCH', 'Field': 'ServiceCode', 'Value': 'AmazonEC2'},
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_name},
                    {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': operating_system},
                    {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                    {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
                    {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'}
                ]
                
                response = client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=filters,
                    MaxResults=1
                )
                
                if not response['PriceList']:
                    logger.warning("No pricing found for EC2 instance",
                                 instance_type=instance_type,
                                 region=region)
                    return None
                
                # Parse pricing data
                price_item = json.loads(response['PriceList'][0])
                on_demand_terms = price_item['terms']['OnDemand']
                
                # Get the first (and usually only) price dimension
                for term_key, term_value in on_demand_terms.items():
                    for dimension_key, dimension_value in term_value['priceDimensions'].items():
                        price_per_unit = dimension_value['pricePerUnit']['USD']
                        return Decimal(price_per_unit)
                
                return None
            
            loop = asyncio.get_event_loop()
            price = await loop.run_in_executor(self.executor, _fetch_price)
            
            if price:
                # Cache the result
                self.cache.set(cache_key, {'price': float(price)})
                logger.info("Fetched EC2 on-demand pricing",
                          instance_type=instance_type,
                          region=region,
                          price=float(price))
            
            return price
            
        except Exception as e:
            logger.error("Failed to fetch EC2 pricing",
                        instance_type=instance_type,
                        region=region,
                        error=str(e))
            return None
    
    async def get_ec2_reserved_pricing(self,
                                      instance_type: str,
                                      region: str,
                                      term_years: int = 1,
                                      payment_option: str = 'No Upfront',
                                      operating_system: str = 'Linux') -> Optional[Dict[str, Decimal]]:
        """
        Get EC2 Reserved Instance pricing.
        
        Args:
            instance_type: EC2 instance type
            region: AWS region
            term_years: 1 or 3 year term
            payment_option: 'No Upfront', 'Partial Upfront', or 'All Upfront'
            operating_system: OS type
            
        Returns:
            Dict with 'upfront' and 'hourly' prices or None
        """
        cache_key = f"ec2_ri_{instance_type}_{region}_{term_years}yr_{payment_option}_{operating_system}"
        
        # Check cache
        cached_price = self.cache.get(cache_key)
        if cached_price:
            logger.debug("Using cached EC2 RI pricing",
                        instance_type=instance_type,
                        term=term_years)
            return {
                'upfront': Decimal(str(cached_price['upfront'])),
                'hourly': Decimal(str(cached_price['hourly']))
            }
        
        try:
            def _fetch_ri_price():
                client = self._get_pricing_client()
                region_name = self._get_region_name(region)
                
                filters = [
                    {'Type': 'TERM_MATCH', 'Field': 'ServiceCode', 'Value': 'AmazonEC2'},
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_name},
                    {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': operating_system},
                    {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                    {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'}
                ]
                
                response = client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=filters,
                    MaxResults=1
                )
                
                if not response['PriceList']:
                    return None
                
                price_item = json.loads(response['PriceList'][0])
                
                # Look for Reserved terms
                if 'terms' not in price_item or 'Reserved' not in price_item['terms']:
                    return None
                
                reserved_terms = price_item['terms']['Reserved']
                
                # Find matching term
                term_length = f"{term_years}yr"
                purchase_option = payment_option.replace(' ', '')
                
                for term_key, term_value in reserved_terms.items():
                    term_attrs = term_value.get('termAttributes', {})
                    
                    if (term_attrs.get('LeaseContractLength') == term_length and
                        term_attrs.get('PurchaseOption') == purchase_option):
                        
                        upfront_price = Decimal('0')
                        hourly_price = Decimal('0')
                        
                        for dimension_key, dimension_value in term_value['priceDimensions'].items():
                            unit = dimension_value.get('unit', '')
                            price = Decimal(dimension_value['pricePerUnit']['USD'])
                            
                            if unit == 'Quantity':  # Upfront
                                upfront_price = price
                            elif unit == 'Hrs':  # Hourly
                                hourly_price = price
                        
                        return {
                            'upfront': upfront_price,
                            'hourly': hourly_price
                        }
                
                return None
            
            loop = asyncio.get_event_loop()
            pricing = await loop.run_in_executor(self.executor, _fetch_ri_price)
            
            if pricing:
                # Cache the result
                self.cache.set(cache_key, {
                    'upfront': float(pricing['upfront']),
                    'hourly': float(pricing['hourly'])
                })
                logger.info("Fetched EC2 RI pricing",
                          instance_type=instance_type,
                          term=term_years,
                          payment=payment_option,
                          upfront=float(pricing['upfront']),
                          hourly=float(pricing['hourly']))
            
            return pricing
            
        except Exception as e:
            logger.error("Failed to fetch EC2 RI pricing",
                        instance_type=instance_type,
                        error=str(e))
            return None
    
    async def get_rds_pricing(self,
                             instance_class: str,
                             region: str,
                             engine: str = 'MySQL',
                             deployment_option: str = 'Single-AZ') -> Optional[Decimal]:
        """
        Get RDS on-demand hourly pricing.
        
        Args:
            instance_class: RDS instance class (e.g., 'db.m5.large')
            region: AWS region
            engine: Database engine (MySQL, PostgreSQL, etc.)
            deployment_option: 'Single-AZ' or 'Multi-AZ'
            
        Returns:
            Hourly price as Decimal or None
        """
        cache_key = f"rds_{instance_class}_{region}_{engine}_{deployment_option}"
        
        cached_price = self.cache.get(cache_key)
        if cached_price:
            return Decimal(str(cached_price['price']))
        
        try:
            def _fetch_rds_price():
                client = self._get_pricing_client()
                region_name = self._get_region_name(region)
                
                filters = [
                    {'Type': 'TERM_MATCH', 'Field': 'ServiceCode', 'Value': 'AmazonRDS'},
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_class},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_name},
                    {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': engine},
                    {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': deployment_option}
                ]
                
                response = client.get_products(
                    ServiceCode='AmazonRDS',
                    Filters=filters,
                    MaxResults=1
                )
                
                if not response['PriceList']:
                    return None
                
                price_item = json.loads(response['PriceList'][0])
                on_demand_terms = price_item['terms']['OnDemand']
                
                for term_key, term_value in on_demand_terms.items():
                    for dimension_key, dimension_value in term_value['priceDimensions'].items():
                        price_per_unit = dimension_value['pricePerUnit']['USD']
                        return Decimal(price_per_unit)
                
                return None
            
            loop = asyncio.get_event_loop()
            price = await loop.run_in_executor(self.executor, _fetch_rds_price)
            
            if price:
                self.cache.set(cache_key, {'price': float(price)})
                logger.info("Fetched RDS pricing",
                          instance_class=instance_class,
                          engine=engine,
                          price=float(price))
            
            return price
            
        except Exception as e:
            logger.error("Failed to fetch RDS pricing",
                        instance_class=instance_class,
                        error=str(e))
            return None
    
    def _get_region_name(self, region_code: str) -> str:
        """Convert region code to region name for pricing API"""
        region_mapping = {
            'us-east-1': 'US East (N. Virginia)',
            'us-east-2': 'US East (Ohio)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'EU (Ireland)',
            'eu-west-2': 'EU (London)',
            'eu-west-3': 'EU (Paris)',
            'eu-central-1': 'EU (Frankfurt)',
            'ap-south-1': 'Asia Pacific (Mumbai)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-southeast-2': 'Asia Pacific (Sydney)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)',
            'ap-northeast-2': 'Asia Pacific (Seoul)',
            'sa-east-1': 'South America (Sao Paulo)',
            'ca-central-1': 'Canada (Central)',
        }
        return region_mapping.get(region_code, region_code)
    
    async def bulk_fetch_ec2_pricing(self, 
                                    instance_types: List[str],
                                    region: str) -> Dict[str, Decimal]:
        """
        Fetch pricing for multiple instance types in parallel.
        
        Args:
            instance_types: List of instance types
            region: AWS region
            
        Returns:
            Dict mapping instance_type to price
        """
        tasks = [
            self.get_ec2_on_demand_pricing(instance_type, region)
            for instance_type in instance_types
        ]
        
        prices = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for instance_type, price in zip(instance_types, prices):
            if isinstance(price, Decimal):
                result[instance_type] = price
            else:
                logger.warning("Failed to fetch price for instance",
                             instance_type=instance_type,
                             error=str(price) if isinstance(price, Exception) else "Unknown")
        
        return result
    
    def clear_cache(self):
        """Clear all cached pricing data"""
        self.cache.clear()
        logger.info("Pricing cache cleared")


# Global pricing service instance
_pricing_service = None

def get_pricing_service() -> AWSPricingService:
    """Get global pricing service instance"""
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = AWSPricingService()
    return _pricing_service
