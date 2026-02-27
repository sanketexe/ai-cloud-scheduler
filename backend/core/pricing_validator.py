"""
Pricing Data Validator

Validates pricing data for consistency, accuracy, and detects anomalies
in multi-cloud pricing information.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from enum import Enum
import statistics
import structlog

logger = structlog.get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of pricing anomalies"""
    PRICE_SPIKE = "price_spike"
    PRICE_DROP = "price_drop"
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    OUTLIER = "outlier"
    INCONSISTENT_CURRENCY = "inconsistent_currency"
    STALE_DATA = "stale_data"


@dataclass
class ValidationError:
    """Validation error details"""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None


@dataclass
class ValidationResult:
    """Result of pricing data validation"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """Pricing anomaly details"""
    type: AnomalyType
    severity: ValidationSeverity
    description: str
    current_value: Any
    expected_range: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PricingDataValidator:
    """
    Comprehensive validator for multi-cloud pricing data.
    
    Validates data format, consistency, and detects anomalies
    based on historical patterns and cross-provider comparisons.
    """
    
    def __init__(self):
        self.supported_providers = {'aws', 'gcp', 'azure'}
        self.supported_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD'}
        self.supported_pricing_units = {
            'hour', 'month', 'year', 'gb', 'tb', 'request', 'invocation',
            'gb-month', 'gb-hour', 'vcpu-hour', 'instance-hour'
        }
        
        # Anomaly detection thresholds
        self.price_spike_threshold = 2.0  # 200% increase
        self.price_drop_threshold = 0.5   # 50% decrease
        self.outlier_std_threshold = 2.5  # Standard deviations
        self.stale_data_hours = 48        # Hours before data is considered stale
    
    def validate_pricing_data(self, pricing_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate pricing data structure and values.
        
        Args:
            pricing_data: Dictionary containing pricing information
            
        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Required fields validation
            self._validate_required_fields(pricing_data, result)
            
            # Data type validation
            self._validate_data_types(pricing_data, result)
            
            # Value range validation
            self._validate_value_ranges(pricing_data, result)
            
            # Business logic validation
            self._validate_business_rules(pricing_data, result)
            
            # Cross-field validation
            self._validate_cross_fields(pricing_data, result)
            
            # Set overall validation status
            result.is_valid = len([e for e in result.errors if e.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0
            
            # Add metadata
            result.metadata = {
                'validation_timestamp': datetime.utcnow().isoformat(),
                'total_errors': len(result.errors),
                'total_warnings': len(result.warnings),
                'provider': pricing_data.get('provider'),
                'service': pricing_data.get('service_name')
            }
            
        except Exception as e:
            logger.error("Validation failed with exception", error=str(e))
            result.is_valid = False
            result.errors.append(ValidationError(
                field='validation',
                message=f"Validation process failed: {str(e)}",
                severity=ValidationSeverity.CRITICAL
            ))
        
        return result
    
    def detect_anomalies(self, 
                        current_data: Dict[str, Any], 
                        historical_data: List[Dict[str, Any]]) -> List[Anomaly]:
        """
        Detect pricing anomalies based on historical patterns.
        
        Args:
            current_data: Current pricing data point
            historical_data: List of historical pricing data points
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            if not historical_data:
                logger.warning("No historical data available for anomaly detection")
                return anomalies
            
            # Price change anomalies
            anomalies.extend(self._detect_price_changes(current_data, historical_data))
            
            # Statistical outliers
            anomalies.extend(self._detect_statistical_outliers(current_data, historical_data))
            
            # Data freshness anomalies
            anomalies.extend(self._detect_stale_data(current_data))
            
            # Currency consistency anomalies
            anomalies.extend(self._detect_currency_inconsistencies(current_data, historical_data))
            
        except Exception as e:
            logger.error("Anomaly detection failed", error=str(e))
            anomalies.append(Anomaly(
                type=AnomalyType.INVALID_FORMAT,
                severity=ValidationSeverity.ERROR,
                description=f"Anomaly detection failed: {str(e)}",
                current_value=current_data
            ))
        
        return anomalies
    
    def validate_batch_pricing_data(self, 
                                   pricing_batch: List[Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """
        Validate a batch of pricing data records.
        
        Args:
            pricing_batch: List of pricing data dictionaries
            
        Returns:
            Dictionary mapping record index to validation result
        """
        results = {}
        
        for i, pricing_data in enumerate(pricing_batch):
            try:
                results[str(i)] = self.validate_pricing_data(pricing_data)
            except Exception as e:
                logger.error("Batch validation failed for record", index=i, error=str(e))
                results[str(i)] = ValidationResult(
                    is_valid=False,
                    errors=[ValidationError(
                        field='batch_validation',
                        message=f"Batch validation failed: {str(e)}",
                        severity=ValidationSeverity.CRITICAL
                    )]
                )
        
        return results
    
    def _validate_required_fields(self, data: Dict[str, Any], result: ValidationResult):
        """Validate required fields are present"""
        required_fields = [
            'provider', 'service_name', 'service_category', 'region',
            'pricing_unit', 'price_per_unit', 'currency', 'effective_date'
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                result.errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing or null",
                    severity=ValidationSeverity.ERROR
                ))
    
    def _validate_data_types(self, data: Dict[str, Any], result: ValidationResult):
        """Validate data types"""
        type_validations = {
            'provider': str,
            'service_name': str,
            'service_category': str,
            'region': str,
            'pricing_unit': str,
            'currency': str
        }
        
        for field, expected_type in type_validations.items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    result.errors.append(ValidationError(
                        field=field,
                        message=f"Field '{field}' must be of type {expected_type.__name__}",
                        severity=ValidationSeverity.ERROR,
                        value=type(data[field]).__name__,
                        expected=expected_type.__name__
                    ))
        
        # Validate price_per_unit as Decimal or convertible to Decimal
        if 'price_per_unit' in data and data['price_per_unit'] is not None:
            try:
                if isinstance(data['price_per_unit'], (int, float, str)):
                    Decimal(str(data['price_per_unit']))
                elif not isinstance(data['price_per_unit'], Decimal):
                    result.errors.append(ValidationError(
                        field='price_per_unit',
                        message="Field 'price_per_unit' must be numeric",
                        severity=ValidationSeverity.ERROR,
                        value=type(data['price_per_unit']).__name__
                    ))
            except (InvalidOperation, ValueError):
                result.errors.append(ValidationError(
                    field='price_per_unit',
                    message="Field 'price_per_unit' is not a valid number",
                    severity=ValidationSeverity.ERROR,
                    value=data['price_per_unit']
                ))
        
        # Validate effective_date as datetime or convertible to datetime
        if 'effective_date' in data and data['effective_date'] is not None:
            if not isinstance(data['effective_date'], datetime):
                try:
                    if isinstance(data['effective_date'], str):
                        datetime.fromisoformat(data['effective_date'].replace('Z', '+00:00'))
                    else:
                        result.errors.append(ValidationError(
                            field='effective_date',
                            message="Field 'effective_date' must be datetime or ISO string",
                            severity=ValidationSeverity.ERROR,
                            value=type(data['effective_date']).__name__
                        ))
                except ValueError:
                    result.errors.append(ValidationError(
                        field='effective_date',
                        message="Field 'effective_date' is not a valid datetime",
                        severity=ValidationSeverity.ERROR,
                        value=data['effective_date']
                    ))
    
    def _validate_value_ranges(self, data: Dict[str, Any], result: ValidationResult):
        """Validate value ranges and constraints"""
        
        # Provider validation
        if 'provider' in data and data['provider']:
            provider = data['provider'].lower()
            if provider not in self.supported_providers:
                result.errors.append(ValidationError(
                    field='provider',
                    message=f"Unsupported provider '{data['provider']}'",
                    severity=ValidationSeverity.ERROR,
                    value=data['provider'],
                    expected=list(self.supported_providers)
                ))
        
        # Currency validation
        if 'currency' in data and data['currency']:
            if data['currency'].upper() not in self.supported_currencies:
                result.warnings.append(ValidationError(
                    field='currency',
                    message=f"Uncommon currency '{data['currency']}'",
                    severity=ValidationSeverity.WARNING,
                    value=data['currency'],
                    expected=list(self.supported_currencies)
                ))
        
        # Pricing unit validation
        if 'pricing_unit' in data and data['pricing_unit']:
            if data['pricing_unit'].lower() not in self.supported_pricing_units:
                result.warnings.append(ValidationError(
                    field='pricing_unit',
                    message=f"Uncommon pricing unit '{data['pricing_unit']}'",
                    severity=ValidationSeverity.WARNING,
                    value=data['pricing_unit'],
                    expected=list(self.supported_pricing_units)
                ))
        
        # Price validation
        if 'price_per_unit' in data and data['price_per_unit'] is not None:
            try:
                price = Decimal(str(data['price_per_unit']))
                if price < 0:
                    result.errors.append(ValidationError(
                        field='price_per_unit',
                        message="Price cannot be negative",
                        severity=ValidationSeverity.ERROR,
                        value=price
                    ))
                elif price > Decimal('10000'):  # Arbitrary high threshold
                    result.warnings.append(ValidationError(
                        field='price_per_unit',
                        message="Price seems unusually high",
                        severity=ValidationSeverity.WARNING,
                        value=price
                    ))
            except (InvalidOperation, ValueError):
                pass  # Already handled in type validation
    
    def _validate_business_rules(self, data: Dict[str, Any], result: ValidationResult):
        """Validate business logic rules"""
        
        # Service category consistency
        if 'service_name' in data and 'service_category' in data:
            service_name = data['service_name'].lower()
            category = data['service_category'].lower()
            
            # Basic service category mapping validation
            compute_services = ['ec2', 'compute engine', 'virtual machines', 'lambda', 'cloud functions', 'azure functions']
            storage_services = ['s3', 'cloud storage', 'blob storage', 'ebs', 'persistent disk']
            database_services = ['rds', 'cloud sql', 'azure sql', 'dynamodb', 'firestore', 'cosmos db']
            
            if category == 'compute' and not any(svc in service_name for svc in compute_services):
                result.warnings.append(ValidationError(
                    field='service_category',
                    message=f"Service '{data['service_name']}' may not belong to 'compute' category",
                    severity=ValidationSeverity.WARNING,
                    value=category
                ))
            elif category == 'storage' and not any(svc in service_name for svc in storage_services):
                result.warnings.append(ValidationError(
                    field='service_category',
                    message=f"Service '{data['service_name']}' may not belong to 'storage' category",
                    severity=ValidationSeverity.WARNING,
                    value=category
                ))
            elif category == 'database' and not any(svc in service_name for svc in database_services):
                result.warnings.append(ValidationError(
                    field='service_category',
                    message=f"Service '{data['service_name']}' may not belong to 'database' category",
                    severity=ValidationSeverity.WARNING,
                    value=category
                ))
        
        # Effective date validation
        if 'effective_date' in data and data['effective_date']:
            try:
                if isinstance(data['effective_date'], str):
                    effective_date = datetime.fromisoformat(data['effective_date'].replace('Z', '+00:00'))
                else:
                    effective_date = data['effective_date']
                
                now = datetime.utcnow()
                if effective_date > now + timedelta(days=1):
                    result.warnings.append(ValidationError(
                        field='effective_date',
                        message="Effective date is in the future",
                        severity=ValidationSeverity.WARNING,
                        value=effective_date.isoformat()
                    ))
                elif effective_date < now - timedelta(days=365):
                    result.warnings.append(ValidationError(
                        field='effective_date',
                        message="Effective date is more than a year old",
                        severity=ValidationSeverity.WARNING,
                        value=effective_date.isoformat()
                    ))
            except (ValueError, AttributeError):
                pass  # Already handled in type validation
    
    def _validate_cross_fields(self, data: Dict[str, Any], result: ValidationResult):
        """Validate relationships between fields"""
        
        # Provider-specific region validation
        if 'provider' in data and 'region' in data:
            provider = data['provider'].lower()
            region = data['region'].lower()
            
            # Basic region validation (simplified)
            aws_regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
            gcp_regions = ['us-central1', 'europe-west1', 'asia-southeast1']
            azure_regions = ['east us', 'west europe', 'southeast asia']
            
            if provider == 'aws' and not any(r in region for r in aws_regions):
                result.warnings.append(ValidationError(
                    field='region',
                    message=f"Region '{data['region']}' may not be valid for AWS",
                    severity=ValidationSeverity.WARNING,
                    value=data['region']
                ))
            elif provider == 'gcp' and not any(r in region for r in gcp_regions):
                result.warnings.append(ValidationError(
                    field='region',
                    message=f"Region '{data['region']}' may not be valid for GCP",
                    severity=ValidationSeverity.WARNING,
                    value=data['region']
                ))
            elif provider == 'azure' and not any(r in region for r in azure_regions):
                result.warnings.append(ValidationError(
                    field='region',
                    message=f"Region '{data['region']}' may not be valid for Azure",
                    severity=ValidationSeverity.WARNING,
                    value=data['region']
                ))
    
    def _detect_price_changes(self, 
                             current_data: Dict[str, Any], 
                             historical_data: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect significant price changes"""
        anomalies = []
        
        if not historical_data:
            return anomalies
        
        try:
            current_price = Decimal(str(current_data.get('price_per_unit', 0)))
            
            # Get the most recent historical price
            historical_data.sort(key=lambda x: x.get('effective_date', datetime.min), reverse=True)
            previous_price = Decimal(str(historical_data[0].get('price_per_unit', 0)))
            
            if previous_price > 0:
                price_ratio = current_price / previous_price
                
                if price_ratio >= self.price_spike_threshold:
                    anomalies.append(Anomaly(
                        type=AnomalyType.PRICE_SPIKE,
                        severity=ValidationSeverity.WARNING,
                        description=f"Price increased by {((price_ratio - 1) * 100):.1f}%",
                        current_value=float(current_price),
                        expected_range={'min': float(previous_price * Decimal('0.9')), 
                                      'max': float(previous_price * Decimal('1.1'))},
                        confidence=0.8,
                        metadata={
                            'previous_price': float(previous_price),
                            'change_ratio': float(price_ratio)
                        }
                    ))
                elif price_ratio <= self.price_drop_threshold:
                    anomalies.append(Anomaly(
                        type=AnomalyType.PRICE_DROP,
                        severity=ValidationSeverity.WARNING,
                        description=f"Price decreased by {((1 - price_ratio) * 100):.1f}%",
                        current_value=float(current_price),
                        expected_range={'min': float(previous_price * Decimal('0.9')), 
                                      'max': float(previous_price * Decimal('1.1'))},
                        confidence=0.8,
                        metadata={
                            'previous_price': float(previous_price),
                            'change_ratio': float(price_ratio)
                        }
                    ))
        
        except (InvalidOperation, ValueError, KeyError) as e:
            logger.warning("Failed to detect price changes", error=str(e))
        
        return anomalies
    
    def _detect_statistical_outliers(self, 
                                   current_data: Dict[str, Any], 
                                   historical_data: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect statistical outliers in pricing data"""
        anomalies = []
        
        if len(historical_data) < 3:  # Need minimum data points for statistics
            return anomalies
        
        try:
            current_price = float(current_data.get('price_per_unit', 0))
            historical_prices = [float(d.get('price_per_unit', 0)) for d in historical_data if d.get('price_per_unit')]
            
            if len(historical_prices) < 3:
                return anomalies
            
            mean_price = statistics.mean(historical_prices)
            std_price = statistics.stdev(historical_prices)
            
            if std_price > 0:
                z_score = abs(current_price - mean_price) / std_price
                
                if z_score > self.outlier_std_threshold:
                    anomalies.append(Anomaly(
                        type=AnomalyType.OUTLIER,
                        severity=ValidationSeverity.WARNING,
                        description=f"Price is {z_score:.1f} standard deviations from historical mean",
                        current_value=current_price,
                        expected_range={'mean': mean_price, 'std': std_price},
                        confidence=min(0.9, z_score / 5.0),  # Higher z-score = higher confidence
                        metadata={
                            'z_score': z_score,
                            'historical_mean': mean_price,
                            'historical_std': std_price,
                            'sample_size': len(historical_prices)
                        }
                    ))
        
        except (ValueError, statistics.StatisticsError) as e:
            logger.warning("Failed to detect statistical outliers", error=str(e))
        
        return anomalies
    
    def _detect_stale_data(self, current_data: Dict[str, Any]) -> List[Anomaly]:
        """Detect stale pricing data"""
        anomalies = []
        
        try:
            if 'last_updated' in current_data and current_data['last_updated']:
                if isinstance(current_data['last_updated'], str):
                    last_updated = datetime.fromisoformat(current_data['last_updated'].replace('Z', '+00:00'))
                else:
                    last_updated = current_data['last_updated']
                
                hours_old = (datetime.utcnow() - last_updated).total_seconds() / 3600
                
                if hours_old > self.stale_data_hours:
                    anomalies.append(Anomaly(
                        type=AnomalyType.STALE_DATA,
                        severity=ValidationSeverity.WARNING,
                        description=f"Pricing data is {hours_old:.1f} hours old",
                        current_value=last_updated.isoformat(),
                        confidence=0.9,
                        metadata={
                            'hours_old': hours_old,
                            'threshold_hours': self.stale_data_hours
                        }
                    ))
        
        except (ValueError, AttributeError) as e:
            logger.warning("Failed to detect stale data", error=str(e))
        
        return anomalies
    
    def _detect_currency_inconsistencies(self, 
                                       current_data: Dict[str, Any], 
                                       historical_data: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect currency inconsistencies"""
        anomalies = []
        
        try:
            current_currency = current_data.get('currency', '').upper()
            historical_currencies = set(d.get('currency', '').upper() for d in historical_data if d.get('currency'))
            
            if historical_currencies and current_currency not in historical_currencies:
                anomalies.append(Anomaly(
                    type=AnomalyType.INCONSISTENT_CURRENCY,
                    severity=ValidationSeverity.WARNING,
                    description=f"Currency '{current_currency}' differs from historical currencies",
                    current_value=current_currency,
                    confidence=0.7,
                    metadata={
                        'historical_currencies': list(historical_currencies)
                    }
                ))
        
        except Exception as e:
            logger.warning("Failed to detect currency inconsistencies", error=str(e))
        
        return anomalies