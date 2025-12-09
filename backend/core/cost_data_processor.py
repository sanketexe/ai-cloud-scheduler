"""
Cost Data Processing Pipeline for FinOps Platform
Handles validation, transformation, and quality checks for cost data
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID
from dataclasses import dataclass
from enum import Enum
import structlog
import hashlib
import json

from .models import CostData, CloudProvider, ProviderType
from .repositories import CostDataRepository, CloudProviderRepository
from .cache_service import CacheService
from .logging_service import LoggingService
from .exceptions import ValidationException, DataProcessingException

logger = structlog.get_logger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DataQualityIssueType(Enum):
    """Types of data quality issues"""
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_DATA_TYPE = "invalid_data_type"
    NEGATIVE_COST = "negative_cost"
    FUTURE_DATE = "future_date"
    DUPLICATE_RECORD = "duplicate_record"
    INVALID_CURRENCY = "invalid_currency"
    MISSING_TAGS = "missing_tags"
    INCONSISTENT_METADATA = "inconsistent_metadata"
    OUTLIER_COST = "outlier_cost"

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    processed_data: Optional[Dict[str, Any]] = None
    
    def add_issue(self, issue_type: DataQualityIssueType, severity: ValidationSeverity, 
                  message: str, field: str = None, value: Any = None):
        """Add a validation issue"""
        issue = {
            'type': issue_type.value,
            'severity': severity.value,
            'message': message,
            'field': field,
            'value': str(value) if value is not None else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.issues.append(issue)
            self.is_valid = False
        else:
            self.warnings.append(issue)

@dataclass
class ProcessingResult:
    """Result of cost data processing"""
    success: bool
    processed_count: int
    created_count: int
    updated_count: int
    skipped_count: int
    error_count: int
    validation_results: List[ValidationResult]
    processing_time: float
    summary: Dict[str, Any]

@dataclass
class DeduplicationResult:
    """Result of duplicate detection"""
    duplicates_found: int
    duplicates_removed: int
    unique_records: List[Dict[str, Any]]
    duplicate_groups: List[List[Dict[str, Any]]]

class CostDataProcessor:
    """Processes and validates cost data from cloud providers"""
    
    # Supported currencies
    SUPPORTED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR'}
    
    # Required fields for cost data
    REQUIRED_FIELDS = {
        'provider_id', 'resource_id', 'resource_type', 'service_name',
        'cost_amount', 'currency', 'cost_date'
    }
    
    # Maximum cost threshold for outlier detection (configurable)
    MAX_DAILY_COST_THRESHOLD = Decimal('100000.00')  # $100,000 per day per resource
    
    def __init__(self, 
                 cost_data_repository: CostDataRepository,
                 cloud_provider_repository: CloudProviderRepository,
                 cache_service: CacheService,
                 logging_service: LoggingService):
        self.cost_data_repo = cost_data_repository
        self.cloud_provider_repo = cloud_provider_repository
        self.cache_service = cache_service
        self.logging_service = logging_service
    
    async def process_cost_data(self, provider_id: UUID, raw_data: List[Dict[str, Any]]) -> ProcessingResult:
        """
        Process raw cost data from cloud providers
        
        Args:
            provider_id: UUID of the cloud provider
            raw_data: List of raw cost data records
            
        Returns:
            ProcessingResult with processing statistics and results
        """
        start_time = datetime.utcnow()
        
        try:
            self.logging_service.info(
                "Starting cost data processing",
                provider_id=str(provider_id),
                record_count=len(raw_data)
            )
            
            # Validate provider exists and is active
            provider = await self.cloud_provider_repo.get_by_id(provider_id)
            if not provider or not provider.is_active:
                raise ValidationException(f"Provider {provider_id} not found or inactive")
            
            # Initialize counters
            processed_count = 0
            created_count = 0
            updated_count = 0
            skipped_count = 0
            error_count = 0
            validation_results = []
            
            # Step 1: Validate and normalize data
            validated_data = []
            for record in raw_data:
                validation_result = await self.validate_cost_data(record, provider)
                validation_results.append(validation_result)
                
                if validation_result.is_valid:
                    validated_data.append(validation_result.processed_data)
                    processed_count += 1
                else:
                    error_count += 1
                    self.logging_service.warning(
                        "Cost data validation failed",
                        provider_id=str(provider_id),
                        resource_id=record.get('resource_id'),
                        issues=validation_result.issues
                    )
            
            # Step 2: Detect and handle duplicates
            dedup_result = await self.detect_duplicates(validated_data)
            unique_data = dedup_result.unique_records
            
            self.logging_service.info(
                "Duplicate detection completed",
                provider_id=str(provider_id),
                duplicates_found=dedup_result.duplicates_found,
                unique_records=len(unique_data)
            )
            
            # Step 3: Save to database
            for record in unique_data:
                try:
                    # Check if record already exists
                    existing = await self._find_existing_record(record)
                    
                    if existing:
                        # Update existing record if data has changed
                        if await self._has_data_changed(existing, record):
                            await self.cost_data_repo.update(existing.id, **record)
                            updated_count += 1
                        else:
                            skipped_count += 1
                    else:
                        # Create new record
                        await self.cost_data_repo.create(**record)
                        created_count += 1
                        
                except Exception as e:
                    error_count += 1
                    self.logging_service.error(
                        "Failed to save cost data record",
                        provider_id=str(provider_id),
                        resource_id=record.get('resource_id'),
                        error=str(e)
                    )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create summary
            summary = {
                'provider_id': str(provider_id),
                'provider_name': provider.name,
                'total_input_records': len(raw_data),
                'validation_passed': processed_count,
                'validation_failed': error_count,
                'duplicates_found': dedup_result.duplicates_found,
                'unique_records': len(unique_data),
                'created': created_count,
                'updated': updated_count,
                'skipped': skipped_count,
                'processing_time_seconds': processing_time
            }
            
            self.logging_service.info(
                "Cost data processing completed",
                **summary
            )
            
            # Cache processing statistics
            await self._cache_processing_stats(provider_id, summary)
            
            return ProcessingResult(
                success=True,
                processed_count=processed_count,
                created_count=created_count,
                updated_count=updated_count,
                skipped_count=skipped_count,
                error_count=error_count,
                validation_results=validation_results,
                processing_time=processing_time,
                summary=summary
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.logging_service.error(
                "Cost data processing failed",
                provider_id=str(provider_id),
                error=str(e),
                processing_time=processing_time
            )
            
            return ProcessingResult(
                success=False,
                processed_count=0,
                created_count=0,
                updated_count=0,
                skipped_count=0,
                error_count=len(raw_data),
                validation_results=[],
                processing_time=processing_time,
                summary={'error': str(e)}
            )
    
    async def validate_cost_data(self, raw_record: Dict[str, Any], provider: CloudProvider) -> ValidationResult:
        """
        Validate and normalize a single cost data record
        
        Args:
            raw_record: Raw cost data record
            provider: Cloud provider information
            
        Returns:
            ValidationResult with validation status and processed data
        """
        result = ValidationResult(is_valid=True, issues=[], warnings=[])
        processed_data = raw_record.copy()
        
        try:
            # Check required fields
            missing_fields = self.REQUIRED_FIELDS - set(raw_record.keys())
            if missing_fields:
                for field in missing_fields:
                    result.add_issue(
                        DataQualityIssueType.MISSING_REQUIRED_FIELD,
                        ValidationSeverity.ERROR,
                        f"Required field '{field}' is missing",
                        field=field
                    )
            
            # Validate provider_id
            if 'provider_id' in raw_record:
                try:
                    provider_uuid = UUID(str(raw_record['provider_id']))
                    if provider_uuid != provider.id:
                        result.add_issue(
                            DataQualityIssueType.INVALID_DATA_TYPE,
                            ValidationSeverity.ERROR,
                            f"Provider ID mismatch: expected {provider.id}, got {provider_uuid}",
                            field='provider_id',
                            value=raw_record['provider_id']
                        )
                    processed_data['provider_id'] = provider_uuid
                except (ValueError, TypeError):
                    result.add_issue(
                        DataQualityIssueType.INVALID_DATA_TYPE,
                        ValidationSeverity.ERROR,
                        "Invalid provider_id format",
                        field='provider_id',
                        value=raw_record['provider_id']
                    )
            
            # Validate cost_amount
            if 'cost_amount' in raw_record:
                try:
                    cost_amount = Decimal(str(raw_record['cost_amount']))
                    if cost_amount < 0:
                        result.add_issue(
                            DataQualityIssueType.NEGATIVE_COST,
                            ValidationSeverity.WARNING,
                            "Negative cost amount detected",
                            field='cost_amount',
                            value=cost_amount
                        )
                    elif cost_amount > self.MAX_DAILY_COST_THRESHOLD:
                        result.add_issue(
                            DataQualityIssueType.OUTLIER_COST,
                            ValidationSeverity.WARNING,
                            f"Cost amount exceeds threshold: {cost_amount}",
                            field='cost_amount',
                            value=cost_amount
                        )
                    processed_data['cost_amount'] = cost_amount
                except (InvalidOperation, ValueError, TypeError):
                    result.add_issue(
                        DataQualityIssueType.INVALID_DATA_TYPE,
                        ValidationSeverity.ERROR,
                        "Invalid cost_amount format",
                        field='cost_amount',
                        value=raw_record['cost_amount']
                    )
            
            # Validate currency
            if 'currency' in raw_record:
                currency = str(raw_record['currency']).upper()
                if currency not in self.SUPPORTED_CURRENCIES:
                    result.add_issue(
                        DataQualityIssueType.INVALID_CURRENCY,
                        ValidationSeverity.WARNING,
                        f"Unsupported currency: {currency}",
                        field='currency',
                        value=currency
                    )
                processed_data['currency'] = currency
            
            # Validate cost_date
            if 'cost_date' in raw_record:
                try:
                    if isinstance(raw_record['cost_date'], str):
                        cost_date = datetime.fromisoformat(raw_record['cost_date'].replace('Z', '+00:00')).date()
                    elif isinstance(raw_record['cost_date'], datetime):
                        cost_date = raw_record['cost_date'].date()
                    elif isinstance(raw_record['cost_date'], date):
                        cost_date = raw_record['cost_date']
                    else:
                        raise ValueError("Invalid date format")
                    
                    # Check if date is in the future
                    if cost_date > date.today():
                        result.add_issue(
                            DataQualityIssueType.FUTURE_DATE,
                            ValidationSeverity.WARNING,
                            f"Cost date is in the future: {cost_date}",
                            field='cost_date',
                            value=cost_date
                        )
                    
                    processed_data['cost_date'] = cost_date
                except (ValueError, TypeError):
                    result.add_issue(
                        DataQualityIssueType.INVALID_DATA_TYPE,
                        ValidationSeverity.ERROR,
                        "Invalid cost_date format",
                        field='cost_date',
                        value=raw_record['cost_date']
                    )
            
            # Validate usage_quantity if present
            if 'usage_quantity' in raw_record and raw_record['usage_quantity'] is not None:
                try:
                    usage_quantity = Decimal(str(raw_record['usage_quantity']))
                    if usage_quantity < 0:
                        result.add_issue(
                            DataQualityIssueType.INVALID_DATA_TYPE,
                            ValidationSeverity.WARNING,
                            "Negative usage quantity",
                            field='usage_quantity',
                            value=usage_quantity
                        )
                    processed_data['usage_quantity'] = usage_quantity
                except (InvalidOperation, ValueError, TypeError):
                    result.add_issue(
                        DataQualityIssueType.INVALID_DATA_TYPE,
                        ValidationSeverity.WARNING,
                        "Invalid usage_quantity format",
                        field='usage_quantity',
                        value=raw_record['usage_quantity']
                    )
            
            # Validate and normalize string fields
            string_fields = ['resource_id', 'resource_type', 'service_name', 'usage_unit']
            for field in string_fields:
                if field in raw_record and raw_record[field] is not None:
                    processed_data[field] = str(raw_record[field]).strip()
                    if not processed_data[field] and field in self.REQUIRED_FIELDS:
                        result.add_issue(
                            DataQualityIssueType.MISSING_REQUIRED_FIELD,
                            ValidationSeverity.ERROR,
                            f"Field '{field}' is empty",
                            field=field
                        )
            
            # Validate and normalize tags
            if 'tags' in raw_record:
                if isinstance(raw_record['tags'], dict):
                    # Normalize tag keys and values
                    normalized_tags = {}
                    for key, value in raw_record['tags'].items():
                        normalized_key = str(key).strip()
                        normalized_value = str(value).strip() if value is not None else ""
                        if normalized_key:
                            normalized_tags[normalized_key] = normalized_value
                    processed_data['tags'] = normalized_tags
                else:
                    result.add_issue(
                        DataQualityIssueType.INVALID_DATA_TYPE,
                        ValidationSeverity.WARNING,
                        "Tags field must be a dictionary",
                        field='tags',
                        value=type(raw_record['tags']).__name__
                    )
                    processed_data['tags'] = {}
            else:
                processed_data['tags'] = {}
            
            # Validate and normalize metadata
            if 'metadata' in raw_record:
                if isinstance(raw_record['metadata'], dict):
                    processed_data['metadata'] = raw_record['metadata']
                else:
                    result.add_issue(
                        DataQualityIssueType.INVALID_DATA_TYPE,
                        ValidationSeverity.WARNING,
                        "Metadata field must be a dictionary",
                        field='metadata',
                        value=type(raw_record['metadata']).__name__
                    )
                    processed_data['metadata'] = {}
            else:
                processed_data['metadata'] = {}
            
            # Add processing metadata
            processed_data['metadata']['processed_at'] = datetime.utcnow().isoformat()
            processed_data['metadata']['processor_version'] = '1.0'
            
            result.processed_data = processed_data
            
        except Exception as e:
            result.add_issue(
                DataQualityIssueType.INCONSISTENT_METADATA,
                ValidationSeverity.CRITICAL,
                f"Validation processing error: {str(e)}",
                value=str(e)
            )
        
        return result
    
    async def detect_duplicates(self, validated_data: List[Dict[str, Any]]) -> DeduplicationResult:
        """
        Detect and remove duplicate records
        
        Args:
            validated_data: List of validated cost data records
            
        Returns:
            DeduplicationResult with deduplication statistics
        """
        if not validated_data:
            return DeduplicationResult(
                duplicates_found=0,
                duplicates_removed=0,
                unique_records=[],
                duplicate_groups=[]
            )
        
        # Create hash-based deduplication
        record_hashes = {}
        duplicate_groups = []
        unique_records = []
        
        for record in validated_data:
            # Create hash based on key fields that identify uniqueness
            hash_fields = {
                'provider_id': str(record.get('provider_id', '')),
                'resource_id': record.get('resource_id', ''),
                'cost_date': str(record.get('cost_date', '')),
                'service_name': record.get('service_name', ''),
                'resource_type': record.get('resource_type', '')
            }
            
            record_hash = hashlib.md5(
                json.dumps(hash_fields, sort_keys=True).encode()
            ).hexdigest()
            
            if record_hash in record_hashes:
                # Duplicate found
                existing_record = record_hashes[record_hash]
                
                # Check if this is the first duplicate for this hash
                duplicate_group = None
                for group in duplicate_groups:
                    if any(self._records_match(r, existing_record) for r in group):
                        duplicate_group = group
                        break
                
                if duplicate_group is None:
                    # Create new duplicate group
                    duplicate_group = [existing_record]
                    duplicate_groups.append(duplicate_group)
                
                duplicate_group.append(record)
                
                # Keep the record with the most recent processing timestamp
                if (record.get('metadata', {}).get('processed_at', '') > 
                    existing_record.get('metadata', {}).get('processed_at', '')):
                    record_hashes[record_hash] = record
                    
            else:
                # Unique record
                record_hashes[record_hash] = record
        
        unique_records = list(record_hashes.values())
        duplicates_found = sum(len(group) - 1 for group in duplicate_groups)
        
        return DeduplicationResult(
            duplicates_found=duplicates_found,
            duplicates_removed=duplicates_found,
            unique_records=unique_records,
            duplicate_groups=duplicate_groups
        )
    
    async def _find_existing_record(self, record: Dict[str, Any]) -> Optional[CostData]:
        """Find existing cost data record in database"""
        try:
            # Query by unique combination of fields
            filters = {
                'provider_id': record['provider_id'],
                'resource_id': record['resource_id'],
                'cost_date': record['cost_date'],
                'service_name': record['service_name']
            }
            
            existing_records = await self.cost_data_repo.get_all(filters=filters, limit=1)
            return existing_records[0] if existing_records else None
            
        except Exception as e:
            self.logging_service.warning(
                "Failed to find existing record",
                resource_id=record.get('resource_id'),
                error=str(e)
            )
            return None
    
    async def _has_data_changed(self, existing: CostData, new_record: Dict[str, Any]) -> bool:
        """Check if cost data has changed compared to existing record"""
        # Compare key fields that might change
        comparable_fields = ['cost_amount', 'usage_quantity', 'tags', 'metadata']
        
        for field in comparable_fields:
            existing_value = getattr(existing, field, None)
            new_value = new_record.get(field)
            
            # Handle Decimal comparison
            if isinstance(existing_value, Decimal) and new_value is not None:
                try:
                    new_decimal = Decimal(str(new_value))
                    if existing_value != new_decimal:
                        return True
                except (InvalidOperation, ValueError):
                    return True
            elif existing_value != new_value:
                return True
        
        return False
    
    def _records_match(self, record1: Dict[str, Any], record2: Dict[str, Any]) -> bool:
        """Check if two records match based on key fields"""
        key_fields = ['provider_id', 'resource_id', 'cost_date', 'service_name']
        
        for field in key_fields:
            if str(record1.get(field, '')) != str(record2.get(field, '')):
                return False
        
        return True
    
    async def _cache_processing_stats(self, provider_id: UUID, summary: Dict[str, Any]) -> None:
        """Cache processing statistics for monitoring"""
        try:
            cache_key = f"cost_processing_stats:{provider_id}:{date.today().isoformat()}"
            await self.cache_service.set(cache_key, summary, ttl=86400)  # 24 hours
        except Exception as e:
            self.logging_service.warning(
                "Failed to cache processing stats",
                provider_id=str(provider_id),
                error=str(e)
            )
    
    async def get_processing_stats(self, provider_id: UUID, days: int = 7) -> List[Dict[str, Any]]:
        """Get processing statistics for the last N days"""
        stats = []
        
        for i in range(days):
            target_date = date.today() - timedelta(days=i)
            cache_key = f"cost_processing_stats:{provider_id}:{target_date.isoformat()}"
            
            try:
                day_stats = await self.cache_service.get(cache_key)
                if day_stats:
                    day_stats['date'] = target_date.isoformat()
                    stats.append(day_stats)
            except Exception as e:
                self.logging_service.warning(
                    "Failed to retrieve processing stats",
                    provider_id=str(provider_id),
                    date=target_date.isoformat(),
                    error=str(e)
                )
        
        return sorted(stats, key=lambda x: x['date'], reverse=True)