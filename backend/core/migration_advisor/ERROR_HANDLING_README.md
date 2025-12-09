# Error Handling and Validation Framework

This document describes the comprehensive error handling, validation, and retry/rollback mechanisms implemented for the Cloud Migration Advisor.

## Overview

The framework consists of three main components:

1. **Error Handler** (`error_handler.py`) - Categorizes errors, provides recovery strategies, and maintains error history
2. **Validation Framework** (`validation.py`) - Validates input data for all migration models with field and cross-field validation
3. **Retry and Rollback** (`retry_rollback.py`) - Provides exponential backoff retry logic and transaction management with rollback

## Error Handler

### Features

- **Error Categorization**: Automatically categorizes errors into assessment, recommendation, migration execution, and resource organization errors
- **Recovery Strategies**: Provides appropriate recovery strategies (retry, rollback, manual intervention, fallback, etc.)
- **Error History**: Maintains a history of all errors with filtering capabilities
- **Detailed Logging**: Logs errors with appropriate severity levels and context

### Usage

```python
from backend.core.migration_advisor import get_error_handler

error_handler = get_error_handler()

try:
    # Your operation
    result = perform_migration_operation()
except Exception as e:
    # Handle the error
    error_response = error_handler.handle_error(
        error=e,
        context={'project_id': 'proj-123', 'phase_id': 'phase-1'}
    )
    
    # Check recovery strategy
    if error_response.can_retry:
        # Retry the operation
        pass
    elif error_response.user_action_required:
        # Notify user
        print(error_response.user_action_message)
```

### Error Categories

- `ASSESSMENT_ERROR` - Errors during assessment phase (incomplete data, validation failures)
- `RECOMMENDATION_ERROR` - Errors during recommendation generation (ML model failures, insufficient data)
- `MIGRATION_EXECUTION_ERROR` - Errors during migration execution (API failures, deployment issues)
- `RESOURCE_ORGANIZATION_ERROR` - Errors during resource organization (tagging conflicts, categorization failures)
- `VALIDATION_ERROR` - Input validation errors
- `PROVIDER_API_ERROR` - Cloud provider API errors
- `DATABASE_ERROR` - Database operation errors
- `CONFIGURATION_ERROR` - Configuration errors

### Error Severity Levels

- `LOW` - Minor issues that don't affect functionality
- `MEDIUM` - Issues that may affect some functionality
- `HIGH` - Significant issues that affect core functionality
- `CRITICAL` - Critical issues that require immediate attention

### Recovery Strategies

- `RETRY` - Automatically retry the operation
- `ROLLBACK` - Rollback changes and restore previous state
- `MANUAL_INTERVENTION` - Requires manual user intervention
- `SKIP` - Skip the failed operation and continue
- `FALLBACK` - Use fallback mechanism (e.g., rule-based recommendations)
- `ABORT` - Abort the entire operation

## Validation Framework

### Features

- **Field Validation**: Validates individual fields (required, type, length, range, format)
- **Cross-Field Validation**: Validates relationships between fields
- **Business Rule Validation**: Enforces business rules and constraints
- **Detailed Error Reporting**: Provides detailed validation errors with field names and constraints
- **Warnings**: Supports validation warnings for non-critical issues

### Usage

```python
from backend.core.migration_advisor import get_validator

validator = get_validator()

# Validate organization profile
data = {
    'company_size': 'LARGE',
    'industry': 'Technology',
    'current_infrastructure': 'ON_PREMISES',
    'it_team_size': 50,
    'cloud_experience_level': 'INTERMEDIATE'
}

result = validator.validate('organization_profile', data)

if result.is_valid:
    # Data is valid, proceed
    pass
else:
    # Handle validation errors
    for error in result.errors:
        print(f"{error.field}: {error.message}")
```

### Supported Data Types

- `organization_profile` - Organization profile data
- `workload_profile` - Workload profile data
- `performance_requirements` - Performance requirements data
- `budget_constraints` - Budget constraints data
- `migration_plan` - Migration plan data
- `organizational_structure` - Organizational structure data
- `categorized_resource` - Categorized resource data

### Validation Rules

#### OrganizationProfile
- `company_size` - Required, must be valid enum (SMALL, MEDIUM, LARGE, ENTERPRISE)
- `industry` - Required, 2-100 characters
- `current_infrastructure` - Required, must be valid enum
- `it_team_size` - Required, must be positive
- `cloud_experience_level` - Required, must be valid enum
- `geographic_presence` - Optional, must be non-empty list if provided

#### WorkloadProfile
- `workload_name` - Required, 1-255 characters
- `application_type` - Required
- `total_compute_cores` - Optional, must be positive
- `total_memory_gb` - Optional, must be positive
- `total_storage_tb` - Optional, must be >= 0

#### PerformanceRequirements
- `availability_target` - Required, 0-100
- `disaster_recovery_rto` - Optional, must be positive
- `disaster_recovery_rpo` - Optional, must be positive
- Cross-field: RTO must be >= RPO

#### BudgetConstraints
- `migration_budget` - Required, must be positive
- `current_monthly_cost` - Optional, must be >= 0
- `target_monthly_cost` - Optional, must be positive
- `acceptable_cost_variance` - Optional, 0-100
- Warning: Target cost > 2x current cost

#### MigrationPlan
- `target_provider` - Required, must be AWS, GCP, or Azure
- `total_duration_days` - Required, must be positive
- `estimated_cost` - Required, must be positive

#### OrganizationalStructure
- `structure_name` - Required, 1-255 characters
- At least one dimension must be defined (teams, projects, environments, regions, cost_centers)

#### CategorizedResource
- `resource_id` - Required
- `resource_type` - Required
- `provider` - Required, must be AWS, GCP, or Azure
- `ownership_status` - Must be valid enum
- Warning: No categorization dimensions set

## Retry and Rollback Mechanisms

### Features

- **Exponential Backoff**: Automatic retry with exponential backoff for transient failures
- **Configurable Retry**: Customizable retry strategies (exponential, linear, fixed delay)
- **Rate Limit Handling**: Special handling for rate limit exceptions
- **Rollback Management**: LIFO stack of rollback actions
- **Transaction Management**: Transaction-like semantics with automatic rollback on failure
- **Decorator Support**: Easy-to-use decorator for automatic retry

### Retry Usage

#### Using Decorator

```python
from backend.core.migration_advisor import with_retry

@with_retry(max_attempts=5, initial_delay=2.0, max_delay=30.0)
def call_cloud_api():
    # API call that may fail transiently
    return api_client.get_resources()

# Automatically retries on CloudProviderException
result = call_cloud_api()
```

#### Using RetryManager

```python
from backend.core.migration_advisor import get_retry_manager, RetryConfig, RetryStrategy

retry_manager = get_retry_manager()

# Configure retry behavior
config = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
)

# Execute with retry
def my_operation():
    return call_external_service()

result = retry_manager.execute_with_retry(
    operation=my_operation,
    operation_name="call_external_service",
    context={'service': 'provider_api'}
)

if result.success:
    print(f"Success after {result.attempts} attempts")
else:
    print(f"Failed after {result.attempts} attempts: {result.last_exception}")
```

### Rollback Usage

#### Using RollbackManager

```python
from backend.core.migration_advisor import RollbackManager

rollback_manager = RollbackManager()

# Register rollback actions as you perform operations
def create_resource():
    resource = api.create_resource()
    
    # Register rollback action
    rollback_manager.register_action(
        action_id=f"delete-{resource.id}",
        description=f"Delete resource {resource.id}",
        rollback_func=api.delete_resource,
        context={'resource_id': resource.id}
    )
    
    return resource

try:
    # Perform operations
    resource1 = create_resource()
    resource2 = create_resource()
    # ... more operations
except Exception as e:
    # Execute rollback (in reverse order)
    rollback_status = rollback_manager.execute_rollback()
    print(f"Rolled back {rollback_status['successful_actions']} actions")
```

#### Using TransactionManager

```python
from backend.core.migration_advisor import get_transaction_manager

transaction_manager = get_transaction_manager()

try:
    # Begin transaction
    transaction_manager.begin_transaction("migrate-resources-123")
    
    # Perform operations and register rollbacks
    for resource in resources:
        migrate_resource(resource)
        
        transaction_manager.register_rollback(
            action_id=f"rollback-{resource.id}",
            description=f"Rollback migration of {resource.name}",
            rollback_func=rollback_migration,
            context={'resource_id': resource.id}
        )
    
    # Commit if all successful
    result = transaction_manager.commit()
    
except Exception as e:
    # Automatic rollback on exception
    result = transaction_manager.rollback()
```

#### Using Context Manager

```python
from backend.core.migration_advisor import TransactionManager

transaction_manager = TransactionManager()

# Automatic commit/rollback with context manager
with transaction_manager:
    transaction_manager.begin_transaction("my-transaction")
    
    # Perform operations
    # ...
    
    # Automatically commits on success
    # Automatically rolls back on exception
```

### Retry Strategies

- `EXPONENTIAL_BACKOFF` - Delay increases exponentially (1s, 2s, 4s, 8s, ...)
- `LINEAR_BACKOFF` - Delay increases linearly (1s, 2s, 3s, 4s, ...)
- `FIXED_DELAY` - Fixed delay between retries (1s, 1s, 1s, ...)
- `NO_RETRY` - No retry, fail immediately

### Retryable Exceptions

By default, the following exceptions are retried:
- `CloudProviderException`
- `CloudProviderRateLimitException`
- `DatabaseException`

## Integration Example

See `error_handling_example.py` for comprehensive examples of using all three components together.

### Complete Workflow Example

```python
from backend.core.migration_advisor import (
    get_error_handler,
    get_validator,
    get_transaction_manager,
    with_retry
)

def create_migration_with_full_error_handling(data, db_session):
    error_handler = get_error_handler()
    validator = get_validator()
    transaction_manager = get_transaction_manager()
    
    try:
        # 1. Validate input
        validation_result = validator.validate('migration_plan', data)
        validation_result.raise_if_invalid()
        
        # 2. Begin transaction
        transaction_manager.begin_transaction("create-migration")
        
        # 3. Call external API with retry
        @with_retry(max_attempts=3)
        def fetch_data():
            return call_provider_api()
        
        provider_data = fetch_data()
        
        # 4. Create resources with rollback
        transaction_manager.register_rollback(
            action_id="cleanup",
            description="Cleanup resources",
            rollback_func=cleanup_resources,
            context={'data': data}
        )
        
        # Create resources...
        
        # 5. Commit transaction
        return transaction_manager.commit()
        
    except Exception as e:
        # Handle error
        error_response = error_handler.handle_error(e, context={'data': data})
        
        # Rollback if needed
        if transaction_manager.in_transaction:
            transaction_manager.rollback()
        
        return {'success': False, 'error': error_response.to_dict()}
```

## Best Practices

1. **Always validate input** before processing
2. **Use retry for transient failures** (network issues, rate limits)
3. **Register rollback actions** for critical operations
4. **Handle errors at appropriate levels** (don't catch and ignore)
5. **Log errors with context** for debugging
6. **Provide user-friendly error messages** for UI
7. **Use transactions for multi-step operations**
8. **Test error scenarios** thoroughly

## Error Codes Reference

### Assessment Errors
- `ASSESSMENT_ERROR` - General assessment error
- `VALIDATION_ERROR` - Input validation failed
- `REQUIRED_FIELD` - Required field missing
- `INVALID_TYPE` - Invalid data type
- `MIN_LENGTH` / `MAX_LENGTH` - String length violation
- `MIN_VALUE` / `MAX_VALUE` - Numeric range violation

### Recommendation Errors
- `RECOMMENDATION_ERROR` - General recommendation error
- `INSUFFICIENT_DATA` - Not enough data for recommendations
- `ML_MODEL_ERROR` - ML model failure

### Migration Errors
- `MIGRATION_EXECUTION_ERROR` - General migration error
- `PROVIDER_API_ERROR` - Cloud provider API error
- `RESOURCE_DEPLOYMENT_FAILED` - Resource deployment failed
- `DEPENDENCY_ERROR` - Dependency resolution failed

### Organization Errors
- `ORGANIZATION_ERROR` - General organization error
- `RESOURCE_NOT_FOUND` - Resource not found
- `TAGGING_CONFLICT` - Tag conflict detected
- `CATEGORIZATION_ERROR` - Categorization failed

## Requirements Coverage

This implementation satisfies the following requirements:

- **Requirement 1.2**: Validation of organizational information
- **Requirement 2.6**: Requirements completeness validation
- **Requirement 4.5**: Rollback procedures for failed migrations
- **Requirement 4.6**: Migration phase validation
- **Requirement 5.4**: Organizational structure validation
- **Requirement 6.6**: Governance validation

## Testing

See test files for comprehensive unit tests:
- Error handler tests
- Validation framework tests
- Retry mechanism tests
- Rollback mechanism tests
- Integration tests
