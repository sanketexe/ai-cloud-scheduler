"""
Migration Advisor Validation Framework

This module provides comprehensive input validation for all migration advisor
data models, including field validation, cross-field validation, and business
rule validation.
"""

import re
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from decimal import Decimal

from ..exceptions import ValidationException, BusinessRuleException
from .models import (
    CompanySize, InfrastructureType, ExperienceLevel,
    MigrationStatus, PhaseStatus, OwnershipStatus
)


class ValidationError:
    """Represents a single validation error"""
    
    def __init__(
        self,
        field: str,
        message: str,
        error_code: str,
        value: Any = None,
        constraint: str = None
    ):
        self.field = field
        self.message = message
        self.error_code = error_code
        self.value = value
        self.constraint = constraint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "field": self.field,
            "message": self.message,
            "error_code": self.error_code,
            "value": self.value,
            "constraint": self.constraint
        }


class ValidationResult:
    """Result of validation operation"""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def add_error(
        self,
        field: str,
        message: str,
        error_code: str,
        value: Any = None,
        constraint: str = None
    ):
        """Add a validation error"""
        self.errors.append(
            ValidationError(field, message, error_code, value, constraint)
        )
    
    def add_warning(
        self,
        field: str,
        message: str,
        error_code: str,
        value: Any = None
    ):
        """Add a validation warning"""
        self.warnings.append(
            ValidationError(field, message, error_code, value)
        )
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings]
        }
    
    def raise_if_invalid(self):
        """Raise ValidationException if validation failed"""
        if not self.is_valid:
            raise ValidationException(
                message="Validation failed",
                validation_errors=self.to_dict()
            )


class FieldValidator:
    """Base field validator"""
    
    @staticmethod
    def required(value: Any, field_name: str, result: ValidationResult):
        """Validate required field"""
        if value is None or (isinstance(value, str) and not value.strip()):
            result.add_error(
                field=field_name,
                message=f"{field_name} is required",
                error_code="REQUIRED_FIELD",
                value=value
            )
    
    @staticmethod
    def string_length(
        value: str,
        field_name: str,
        min_length: int = None,
        max_length: int = None,
        result: ValidationResult = None
    ):
        """Validate string length"""
        if value is None:
            return
        
        if not isinstance(value, str):
            result.add_error(
                field=field_name,
                message=f"{field_name} must be a string",
                error_code="INVALID_TYPE",
                value=value
            )
            return
        
        length = len(value)
        
        if min_length is not None and length < min_length:
            result.add_error(
                field=field_name,
                message=f"{field_name} must be at least {min_length} characters",
                error_code="MIN_LENGTH",
                value=value,
                constraint=f"min_length={min_length}"
            )
        
        if max_length is not None and length > max_length:
            result.add_error(
                field=field_name,
                message=f"{field_name} must be at most {max_length} characters",
                error_code="MAX_LENGTH",
                value=value,
                constraint=f"max_length={max_length}"
            )
    
    @staticmethod
    def numeric_range(
        value: Union[int, float, Decimal],
        field_name: str,
        min_value: Union[int, float, Decimal] = None,
        max_value: Union[int, float, Decimal] = None,
        result: ValidationResult = None
    ):
        """Validate numeric range"""
        if value is None:
            return
        
        if not isinstance(value, (int, float, Decimal)):
            result.add_error(
                field=field_name,
                message=f"{field_name} must be a number",
                error_code="INVALID_TYPE",
                value=value
            )
            return
        
        if min_value is not None and value < min_value:
            result.add_error(
                field=field_name,
                message=f"{field_name} must be at least {min_value}",
                error_code="MIN_VALUE",
                value=value,
                constraint=f"min_value={min_value}"
            )
        
        if max_value is not None and value > max_value:
            result.add_error(
                field=field_name,
                message=f"{field_name} must be at most {max_value}",
                error_code="MAX_VALUE",
                value=value,
                constraint=f"max_value={max_value}"
            )
    
    @staticmethod
    def email(value: str, field_name: str, result: ValidationResult):
        """Validate email format"""
        if value is None:
            return
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            result.add_error(
                field=field_name,
                message=f"{field_name} must be a valid email address",
                error_code="INVALID_EMAIL",
                value=value
            )
    
    @staticmethod
    def enum_value(value: Any, field_name: str, enum_class, result: ValidationResult):
        """Validate enum value"""
        if value is None:
            return
        
        if isinstance(value, enum_class):
            return
        
        valid_values = [e.value for e in enum_class]
        if value not in valid_values:
            result.add_error(
                field=field_name,
                message=f"{field_name} must be one of: {', '.join(valid_values)}",
                error_code="INVALID_ENUM",
                value=value,
                constraint=f"valid_values={valid_values}"
            )
    
    @staticmethod
    def list_not_empty(value: List, field_name: str, result: ValidationResult):
        """Validate list is not empty"""
        if value is None:
            return
        
        if not isinstance(value, list):
            result.add_error(
                field=field_name,
                message=f"{field_name} must be a list",
                error_code="INVALID_TYPE",
                value=value
            )
            return
        
        if len(value) == 0:
            result.add_error(
                field=field_name,
                message=f"{field_name} cannot be empty",
                error_code="EMPTY_LIST",
                value=value
            )
    
    @staticmethod
    def positive_number(value: Union[int, float, Decimal], field_name: str, result: ValidationResult):
        """Validate number is positive"""
        if value is None:
            return
        
        if not isinstance(value, (int, float, Decimal)):
            result.add_error(
                field=field_name,
                message=f"{field_name} must be a number",
                error_code="INVALID_TYPE",
                value=value
            )
            return
        
        if value <= 0:
            result.add_error(
                field=field_name,
                message=f"{field_name} must be positive",
                error_code="NOT_POSITIVE",
                value=value
            )


class OrganizationProfileValidator:
    """Validator for OrganizationProfile data"""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ValidationResult:
        """Validate organization profile data"""
        result = ValidationResult()
        
        # Required fields
        FieldValidator.required(data.get('company_size'), 'company_size', result)
        FieldValidator.required(data.get('industry'), 'industry', result)
        FieldValidator.required(data.get('current_infrastructure'), 'current_infrastructure', result)
        FieldValidator.required(data.get('it_team_size'), 'it_team_size', result)
        FieldValidator.required(data.get('cloud_experience_level'), 'cloud_experience_level', result)
        
        # Enum validations
        FieldValidator.enum_value(
            data.get('company_size'),
            'company_size',
            CompanySize,
            result
        )
        FieldValidator.enum_value(
            data.get('current_infrastructure'),
            'current_infrastructure',
            InfrastructureType,
            result
        )
        FieldValidator.enum_value(
            data.get('cloud_experience_level'),
            'cloud_experience_level',
            ExperienceLevel,
            result
        )
        
        # String validations
        FieldValidator.string_length(
            data.get('industry'),
            'industry',
            min_length=2,
            max_length=100,
            result=result
        )
        
        # Numeric validations
        FieldValidator.positive_number(
            data.get('it_team_size'),
            'it_team_size',
            result
        )
        
        # List validations
        if data.get('geographic_presence') is not None:
            FieldValidator.list_not_empty(
                data.get('geographic_presence'),
                'geographic_presence',
                result
            )
        
        return result


class WorkloadProfileValidator:
    """Validator for WorkloadProfile data"""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ValidationResult:
        """Validate workload profile data"""
        result = ValidationResult()
        
        # Required fields
        FieldValidator.required(data.get('workload_name'), 'workload_name', result)
        FieldValidator.required(data.get('application_type'), 'application_type', result)
        
        # String validations
        FieldValidator.string_length(
            data.get('workload_name'),
            'workload_name',
            min_length=1,
            max_length=255,
            result=result
        )
        
        # Numeric validations
        if data.get('total_compute_cores') is not None:
            FieldValidator.positive_number(
                data.get('total_compute_cores'),
                'total_compute_cores',
                result
            )
        
        if data.get('total_memory_gb') is not None:
            FieldValidator.positive_number(
                data.get('total_memory_gb'),
                'total_memory_gb',
                result
            )
        
        if data.get('total_storage_tb') is not None:
            FieldValidator.numeric_range(
                data.get('total_storage_tb'),
                'total_storage_tb',
                min_value=0,
                result=result
            )
        
        return result


class PerformanceRequirementsValidator:
    """Validator for PerformanceRequirements data"""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ValidationResult:
        """Validate performance requirements data"""
        result = ValidationResult()
        
        # Required fields
        FieldValidator.required(data.get('availability_target'), 'availability_target', result)
        
        # Availability target validation (0-100)
        FieldValidator.numeric_range(
            data.get('availability_target'),
            'availability_target',
            min_value=0,
            max_value=100,
            result=result
        )
        
        # RTO/RPO validations
        if data.get('disaster_recovery_rto') is not None:
            FieldValidator.positive_number(
                data.get('disaster_recovery_rto'),
                'disaster_recovery_rto',
                result
            )
        
        if data.get('disaster_recovery_rpo') is not None:
            FieldValidator.positive_number(
                data.get('disaster_recovery_rpo'),
                'disaster_recovery_rpo',
                result
            )
        
        # Cross-field validation: RTO should be >= RPO
        rto = data.get('disaster_recovery_rto')
        rpo = data.get('disaster_recovery_rpo')
        if rto is not None and rpo is not None and rto < rpo:
            result.add_error(
                field='disaster_recovery_rto',
                message='RTO (Recovery Time Objective) must be greater than or equal to RPO (Recovery Point Objective)',
                error_code='INVALID_RTO_RPO',
                value=rto,
                constraint=f'rto >= rpo ({rpo})'
            )
        
        return result


class BudgetConstraintsValidator:
    """Validator for BudgetConstraints data"""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ValidationResult:
        """Validate budget constraints data"""
        result = ValidationResult()
        
        # Required fields
        FieldValidator.required(data.get('migration_budget'), 'migration_budget', result)
        
        # Budget validations
        FieldValidator.positive_number(
            data.get('migration_budget'),
            'migration_budget',
            result
        )
        
        if data.get('current_monthly_cost') is not None:
            FieldValidator.numeric_range(
                data.get('current_monthly_cost'),
                'current_monthly_cost',
                min_value=0,
                result=result
            )
        
        if data.get('target_monthly_cost') is not None:
            FieldValidator.positive_number(
                data.get('target_monthly_cost'),
                'target_monthly_cost',
                result
            )
        
        # Cross-field validation: target cost should be reasonable
        current_cost = data.get('current_monthly_cost')
        target_cost = data.get('target_monthly_cost')
        
        if current_cost and target_cost and target_cost > current_cost * 2:
            result.add_warning(
                field='target_monthly_cost',
                message='Target monthly cost is more than double the current cost. Please verify this is intentional.',
                error_code='HIGH_TARGET_COST',
                value=target_cost
            )
        
        # Acceptable variance validation
        if data.get('acceptable_cost_variance') is not None:
            FieldValidator.numeric_range(
                data.get('acceptable_cost_variance'),
                'acceptable_cost_variance',
                min_value=0,
                max_value=100,
                result=result
            )
        
        return result


class MigrationPlanValidator:
    """Validator for MigrationPlan data"""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ValidationResult:
        """Validate migration plan data"""
        result = ValidationResult()
        
        # Required fields
        FieldValidator.required(data.get('target_provider'), 'target_provider', result)
        FieldValidator.required(data.get('total_duration_days'), 'total_duration_days', result)
        FieldValidator.required(data.get('estimated_cost'), 'estimated_cost', result)
        
        # Numeric validations
        FieldValidator.positive_number(
            data.get('total_duration_days'),
            'total_duration_days',
            result
        )
        
        FieldValidator.positive_number(
            data.get('estimated_cost'),
            'estimated_cost',
            result
        )
        
        # Provider validation
        valid_providers = ['AWS', 'GCP', 'Azure']
        if data.get('target_provider') and data.get('target_provider') not in valid_providers:
            result.add_error(
                field='target_provider',
                message=f"target_provider must be one of: {', '.join(valid_providers)}",
                error_code='INVALID_PROVIDER',
                value=data.get('target_provider')
            )
        
        return result


class OrganizationalStructureValidator:
    """Validator for OrganizationalStructure data"""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ValidationResult:
        """Validate organizational structure data"""
        result = ValidationResult()
        
        # Required fields
        FieldValidator.required(data.get('structure_name'), 'structure_name', result)
        
        # String validations
        FieldValidator.string_length(
            data.get('structure_name'),
            'structure_name',
            min_length=1,
            max_length=255,
            result=result
        )
        
        # At least one dimension should be defined
        has_dimension = any([
            data.get('teams'),
            data.get('projects'),
            data.get('environments'),
            data.get('regions'),
            data.get('cost_centers')
        ])
        
        if not has_dimension:
            result.add_error(
                field='organizational_structure',
                message='At least one organizational dimension (teams, projects, environments, regions, or cost_centers) must be defined',
                error_code='NO_DIMENSIONS',
                value=None
            )
        
        return result


class CategorizedResourceValidator:
    """Validator for CategorizedResource data"""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ValidationResult:
        """Validate categorized resource data"""
        result = ValidationResult()
        
        # Required fields
        FieldValidator.required(data.get('resource_id'), 'resource_id', result)
        FieldValidator.required(data.get('resource_type'), 'resource_type', result)
        FieldValidator.required(data.get('provider'), 'provider', result)
        
        # Provider validation
        valid_providers = ['AWS', 'GCP', 'Azure']
        if data.get('provider') and data.get('provider') not in valid_providers:
            result.add_error(
                field='provider',
                message=f"provider must be one of: {', '.join(valid_providers)}",
                error_code='INVALID_PROVIDER',
                value=data.get('provider')
            )
        
        # Ownership status validation
        FieldValidator.enum_value(
            data.get('ownership_status'),
            'ownership_status',
            OwnershipStatus,
            result
        )
        
        # At least one categorization dimension should be set
        has_categorization = any([
            data.get('team'),
            data.get('project'),
            data.get('environment'),
            data.get('region'),
            data.get('cost_center')
        ])
        
        if not has_categorization:
            result.add_warning(
                field='categorization',
                message='Resource has no categorization dimensions set',
                error_code='NO_CATEGORIZATION',
                value=None
            )
        
        return result


class MigrationValidator:
    """
    Main validation orchestrator for migration advisor
    
    Provides validation for all migration-related data models with
    comprehensive field and cross-field validation.
    """
    
    def __init__(self):
        self.validators = {
            'organization_profile': OrganizationProfileValidator,
            'workload_profile': WorkloadProfileValidator,
            'performance_requirements': PerformanceRequirementsValidator,
            'budget_constraints': BudgetConstraintsValidator,
            'migration_plan': MigrationPlanValidator,
            'organizational_structure': OrganizationalStructureValidator,
            'categorized_resource': CategorizedResourceValidator,
        }
    
    def validate(self, data_type: str, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data for a specific type
        
        Args:
            data_type: Type of data to validate
            data: Data dictionary to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        validator = self.validators.get(data_type)
        
        if not validator:
            result = ValidationResult()
            result.add_error(
                field='data_type',
                message=f"Unknown data type: {data_type}",
                error_code='UNKNOWN_DATA_TYPE',
                value=data_type
            )
            return result
        
        return validator.validate(data)
    
    def validate_and_raise(self, data_type: str, data: Dict[str, Any]):
        """
        Validate data and raise exception if invalid
        
        Args:
            data_type: Type of data to validate
            data: Data dictionary to validate
            
        Raises:
            ValidationException: If validation fails
        """
        result = self.validate(data_type, data)
        result.raise_if_invalid()


# Global validator instance
_validator = MigrationValidator()


def get_validator() -> MigrationValidator:
    """Get the global validator instance"""
    return _validator
