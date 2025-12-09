"""
Example usage of error handling, validation, and retry/rollback mechanisms

This module demonstrates how to use the error handling framework,
validation framework, and retry/rollback mechanisms together.
"""

from typing import Dict, Any
from sqlalchemy.orm import Session

from .error_handler import get_error_handler, AssessmentError
from .validation import get_validator
from .retry_rollback import (
    get_retry_manager,
    get_transaction_manager,
    with_retry,
    RetryConfig,
    RetryStrategy,
)
from ..exceptions import ValidationException, CloudProviderException


# Example 1: Validation with error handling
def validate_and_create_organization_profile(
    data: Dict[str, Any],
    db_session: Session
) -> Dict[str, Any]:
    """
    Example: Validate organization profile data and handle errors
    """
    error_handler = get_error_handler()
    validator = get_validator()
    
    try:
        # Validate the data
        validation_result = validator.validate('organization_profile', data)
        
        # Check if validation passed
        if not validation_result.is_valid:
            # Create assessment error from validation errors
            raise ValidationException(
                message="Organization profile validation failed",
                validation_errors=validation_result.to_dict()
            )
        
        # If validation passed, create the profile
        # (In real implementation, this would create database record)
        return {
            'success': True,
            'message': 'Organization profile created successfully',
            'data': data
        }
    
    except Exception as e:
        # Handle the error
        error_response = error_handler.handle_error(
            error=e,
            context={'data_type': 'organization_profile'}
        )
        
        return {
            'success': False,
            'error': error_response.to_dict()
        }


# Example 2: Retry with exponential backoff
@with_retry(max_attempts=5, initial_delay=2.0, max_delay=30.0)
def call_cloud_provider_api(provider: str, endpoint: str) -> Dict[str, Any]:
    """
    Example: Call cloud provider API with automatic retry
    
    The @with_retry decorator automatically retries on CloudProviderException
    with exponential backoff.
    """
    # Simulated API call
    # In real implementation, this would make actual API call
    
    # If API call fails, it will be automatically retried
    # raise CloudProviderException("API temporarily unavailable", provider_type=provider)
    
    return {
        'provider': provider,
        'endpoint': endpoint,
        'data': {'resources': []}
    }


# Example 3: Transaction with rollback
def migrate_resources_with_rollback(
    resources: list,
    target_provider: str,
    db_session: Session
) -> Dict[str, Any]:
    """
    Example: Migrate resources with automatic rollback on failure
    """
    transaction_manager = get_transaction_manager()
    error_handler = get_error_handler()
    
    try:
        # Begin transaction
        transaction_manager.begin_transaction(
            transaction_id=f"migrate-{target_provider}-{len(resources)}"
        )
        
        migrated_resources = []
        
        for resource in resources:
            # Simulate resource migration
            # In real implementation, this would migrate actual resources
            
            # Register rollback action for this resource
            transaction_manager.register_rollback(
                action_id=f"rollback-{resource['id']}",
                description=f"Rollback migration of {resource['name']}",
                rollback_func=rollback_resource_migration,
                context={'resource_id': resource['id'], 'provider': target_provider}
            )
            
            migrated_resources.append(resource)
        
        # Commit transaction if all successful
        result = transaction_manager.commit()
        
        return {
            'success': True,
            'migrated_count': len(migrated_resources),
            'transaction': result
        }
    
    except Exception as e:
        # Handle error and rollback
        error_response = error_handler.handle_error(
            error=e,
            context={'target_provider': target_provider, 'resource_count': len(resources)}
        )
        
        # Rollback transaction
        rollback_result = transaction_manager.rollback()
        
        return {
            'success': False,
            'error': error_response.to_dict(),
            'rollback': rollback_result
        }


def rollback_resource_migration(resource_id: str, provider: str):
    """
    Rollback function for resource migration
    
    This would be called automatically if the transaction fails.
    """
    # In real implementation, this would:
    # 1. Delete the resource from target provider
    # 2. Restore any state changes
    # 3. Update database records
    pass


# Example 4: Combined usage - validation, retry, and error handling
def create_migration_plan_with_full_error_handling(
    plan_data: Dict[str, Any],
    db_session: Session
) -> Dict[str, Any]:
    """
    Example: Create migration plan with full error handling
    
    Demonstrates:
    - Input validation
    - Retry logic for external calls
    - Transaction management
    - Comprehensive error handling
    """
    error_handler = get_error_handler()
    validator = get_validator()
    retry_manager = get_retry_manager()
    transaction_manager = get_transaction_manager()
    
    try:
        # Step 1: Validate input
        validation_result = validator.validate('migration_plan', plan_data)
        validation_result.raise_if_invalid()
        
        # Step 2: Begin transaction
        transaction_manager.begin_transaction(
            transaction_id=f"create-plan-{plan_data.get('plan_id')}"
        )
        
        # Step 3: Call external APIs with retry
        def fetch_provider_data():
            return call_cloud_provider_api(
                provider=plan_data['target_provider'],
                endpoint='/pricing'
            )
        
        retry_result = retry_manager.execute_with_retry(
            operation=fetch_provider_data,
            operation_name="fetch_provider_pricing",
            context={'provider': plan_data['target_provider']}
        )
        
        if not retry_result.success:
            raise retry_result.last_exception
        
        # Step 4: Create plan (with rollback registered)
        transaction_manager.register_rollback(
            action_id="delete-plan",
            description="Delete migration plan",
            rollback_func=delete_migration_plan,
            context={'plan_id': plan_data.get('plan_id')}
        )
        
        # Simulate plan creation
        # In real implementation, this would create database records
        
        # Step 5: Commit transaction
        transaction_result = transaction_manager.commit()
        
        return {
            'success': True,
            'plan_id': plan_data.get('plan_id'),
            'transaction': transaction_result,
            'retry_attempts': retry_result.attempts
        }
    
    except Exception as e:
        # Handle error
        error_response = error_handler.handle_error(
            error=e,
            context={'plan_data': plan_data}
        )
        
        # Rollback if in transaction
        if transaction_manager.in_transaction:
            rollback_result = transaction_manager.rollback()
            
            return {
                'success': False,
                'error': error_response.to_dict(),
                'rollback': rollback_result
            }
        
        return {
            'success': False,
            'error': error_response.to_dict()
        }


def delete_migration_plan(plan_id: str):
    """Rollback function to delete migration plan"""
    # In real implementation, this would delete the plan from database
    pass


# Example 5: Error history and reporting
def get_project_error_report(project_id: str) -> Dict[str, Any]:
    """
    Example: Get error report for a migration project
    """
    error_handler = get_error_handler()
    
    # Get error history for project
    errors = error_handler.get_error_history(
        project_id=project_id,
        limit=50
    )
    
    # Categorize errors
    error_summary = {
        'total_errors': len(errors),
        'by_category': {},
        'by_severity': {},
        'recent_errors': errors[:10]
    }
    
    for error in errors:
        category = error['category']
        severity = error['severity']
        
        error_summary['by_category'][category] = \
            error_summary['by_category'].get(category, 0) + 1
        error_summary['by_severity'][severity] = \
            error_summary['by_severity'].get(severity, 0) + 1
    
    return error_summary
