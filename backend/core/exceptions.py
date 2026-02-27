"""
Custom exception hierarchy for the FinOps Platform.

This module defines a comprehensive exception hierarchy that provides
structured error handling with error codes, context tracking, and
serialization capabilities for API responses.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json


class FinOpsException(Exception):
    """
    Base exception class for all FinOps Platform errors.
    
    Provides structured error handling with error codes, context tracking,
    and serialization capabilities for consistent API error responses.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: str = None, 
        details: Dict[str, Any] = None,
        correlation_id: str = None,
        user_id: str = None
    ):
        """
        Initialize FinOps exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            details: Additional context and debugging information
            correlation_id: Request correlation ID for tracing
            user_id: User ID associated with the error (if applicable)
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.correlation_id = correlation_id
        self.user_id = user_id
        self.timestamp = datetime.utcnow()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize exception to dictionary for API responses.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
                "timestamp": self.timestamp.isoformat() + "Z",
                "correlation_id": self.correlation_id
            }
        }
    
    def to_json(self) -> str:
        """
        Serialize exception to JSON string.
        
        Returns:
            JSON string representation of the exception
        """
        return json.dumps(self.to_dict(), default=str)
    
    def add_context(self, key: str, value: Any) -> 'FinOpsException':
        """
        Add additional context to the exception.
        
        Args:
            key: Context key
            value: Context value
            
        Returns:
            Self for method chaining
        """
        self.details[key] = value
        return self


class CloudProviderException(FinOpsException):
    """
    Exception for cloud provider API integration errors.
    
    Raised when there are issues with cloud provider API calls,
    authentication, or data retrieval.
    """
    
    def __init__(
        self, 
        message: str, 
        provider_type: str = None,
        provider_id: str = None,
        api_endpoint: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if provider_type:
            details['provider_type'] = provider_type
        if provider_id:
            details['provider_id'] = provider_id
        if api_endpoint:
            details['api_endpoint'] = api_endpoint
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'CLOUD_PROVIDER_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class CloudProviderConnectionException(CloudProviderException):
    """Exception for cloud provider connection failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='CLOUD_PROVIDER_CONNECTION_ERROR',
            **kwargs
        )


class CloudProviderAuthenticationException(CloudProviderException):
    """Exception for cloud provider authentication failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='CLOUD_PROVIDER_AUTH_ERROR',
            **kwargs
        )


class CloudProviderRateLimitException(CloudProviderException):
    """Exception for cloud provider rate limiting."""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after'] = retry_after
            
        super().__init__(
            message=message,
            error_code='CLOUD_PROVIDER_RATE_LIMIT',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class AuthenticationException(FinOpsException):
    """
    Exception for authentication and authorization errors.
    
    Raised when there are issues with user authentication,
    token validation, or authorization checks.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'AUTHENTICATION_ERROR'),
            **kwargs
        )


class InvalidTokenException(AuthenticationException):
    """Exception for invalid or expired tokens."""
    
    def __init__(self, message: str = "Invalid or expired token", **kwargs):
        super().__init__(
            message=message,
            error_code='INVALID_TOKEN',
            **kwargs
        )


class InsufficientPermissionsException(AuthenticationException):
    """Exception for insufficient user permissions."""
    
    def __init__(self, message: str, required_permission: str = None, **kwargs):
        details = kwargs.get('details', {})
        if required_permission:
            details['required_permission'] = required_permission
            
        super().__init__(
            message=message,
            error_code='INSUFFICIENT_PERMISSIONS',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ValidationException(FinOpsException):
    """
    Exception for data validation errors.
    
    Raised when input data fails validation checks,
    schema validation, or business rule validation.
    """
    
    def __init__(
        self, 
        message: str, 
        field: str = None,
        validation_errors: Dict[str, Any] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if validation_errors:
            details['validation_errors'] = validation_errors
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'VALIDATION_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class SchemaValidationException(ValidationException):
    """Exception for schema validation failures."""
    
    def __init__(self, message: str, schema_errors: list = None, **kwargs):
        details = kwargs.get('details', {})
        if schema_errors:
            details['schema_errors'] = schema_errors
            
        super().__init__(
            message=message,
            error_code='SCHEMA_VALIDATION_ERROR',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class BusinessRuleException(ValidationException):
    """Exception for business rule violations."""
    
    def __init__(self, message: str, rule_name: str = None, **kwargs):
        details = kwargs.get('details', {})
        if rule_name:
            details['rule_name'] = rule_name
            
        super().__init__(
            message=message,
            error_code='BUSINESS_RULE_VIOLATION',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class DatabaseException(FinOpsException):
    """
    Exception for database operation errors.
    
    Raised when there are issues with database connections,
    queries, transactions, or data integrity.
    """
    
    def __init__(
        self, 
        message: str, 
        operation: str = None,
        table: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if table:
            details['table'] = table
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'DATABASE_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class DatabaseConnectionException(DatabaseException):
    """Exception for database connection failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='DATABASE_CONNECTION_ERROR',
            **kwargs
        )


class DatabaseIntegrityException(DatabaseException):
    """Exception for database integrity constraint violations."""
    
    def __init__(self, message: str, constraint: str = None, **kwargs):
        details = kwargs.get('details', {})
        if constraint:
            details['constraint'] = constraint
            
        super().__init__(
            message=message,
            error_code='DATABASE_INTEGRITY_ERROR',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class CacheException(FinOpsException):
    """
    Exception for cache operation errors.
    
    Raised when there are issues with cache connections,
    operations, or data serialization.
    """
    
    def __init__(
        self, 
        message: str, 
        cache_key: str = None,
        operation: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if cache_key:
            details['cache_key'] = cache_key
        if operation:
            details['operation'] = operation
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'CACHE_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class CacheConnectionException(CacheException):
    """Exception for cache connection failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='CACHE_CONNECTION_ERROR',
            **kwargs
        )


class ConfigurationException(FinOpsException):
    """
    Exception for configuration errors.
    
    Raised when there are issues with application configuration,
    missing environment variables, or invalid settings.
    """
    
    def __init__(
        self, 
        message: str, 
        config_key: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'CONFIGURATION_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class ExternalServiceException(FinOpsException):
    """
    Exception for external service integration errors.
    
    Raised when there are issues with external service calls,
    timeouts, or service unavailability.
    """
    
    def __init__(
        self, 
        message: str, 
        service_name: str = None,
        endpoint: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        if endpoint:
            details['endpoint'] = endpoint
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'EXTERNAL_SERVICE_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class RateLimitException(FinOpsException):
    """
    Exception for rate limiting errors.
    
    Raised when API rate limits are exceeded.
    """
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        retry_after: int = None,
        limit: int = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after'] = retry_after
        if limit:
            details['limit'] = limit
            
        super().__init__(
            message=message,
            error_code='RATE_LIMIT_EXCEEDED',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ResourceNotFoundException(FinOpsException):
    """
    Exception for resource not found errors.
    
    Raised when requested resources cannot be found.
    """
    
    def __init__(
        self, 
        message: str, 
        resource_type: str = None,
        resource_id: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
            
        super().__init__(
            message=message,
            error_code='RESOURCE_NOT_FOUND',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ConflictException(FinOpsException):
    """
    Exception for resource conflict errors.
    
    Raised when operations conflict with existing resources or state.
    """
    
    def __init__(
        self, 
        message: str, 
        resource_type: str = None,
        conflicting_resource: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if conflicting_resource:
            details['conflicting_resource'] = conflicting_resource
            
        super().__init__(
            message=message,
            error_code='RESOURCE_CONFLICT',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class WebhookException(FinOpsException):
    """
    Exception for webhook system errors.
    
    Raised when there are issues with webhook delivery,
    endpoint configuration, or event processing.
    """
    
    def __init__(
        self, 
        message: str, 
        endpoint_id: str = None,
        event_type: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if endpoint_id:
            details['endpoint_id'] = endpoint_id
        if event_type:
            details['event_type'] = event_type
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'WEBHOOK_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class WebhookDeliveryException(WebhookException):
    """Exception for webhook delivery failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='WEBHOOK_DELIVERY_ERROR',
            **kwargs
        )


class WebhookConfigurationException(WebhookException):
    """Exception for webhook configuration errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='WEBHOOK_CONFIGURATION_ERROR',
            **kwargs
        )


class WebhookSecurityException(WebhookException):
    """Exception for webhook security validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='WEBHOOK_SECURITY_ERROR',
            **kwargs
        )


class NLPProcessingError(FinOpsException):
    """
    Exception for Natural Language Processing errors.
    
    Raised when there are issues with NLP model loading,
    query parsing, intent classification, or response generation.
    """
    
    def __init__(
        self, 
        message: str, 
        model_name: str = None,
        query: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if query:
            details['query'] = query[:100] + "..." if len(query) > 100 else query
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'NLP_PROCESSING_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class AIServiceError(FinOpsException):
    """
    Exception for AI service errors.
    
    Raised when there are issues with AI service operations,
    model loading, inference, or system coordination.
    """
    
    def __init__(
        self, 
        message: str, 
        service_name: str = None,
        operation: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        if operation:
            details['operation'] = operation
            
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'AI_SERVICE_ERROR'),
            details=details,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'error_code']}
        )


class ModelNotFoundError(AIServiceError):
    """Exception for model not found errors."""
    
    def __init__(self, message: str, model_id: str = None, **kwargs):
        details = kwargs.get('details', {})
        if model_id:
            details['model_id'] = model_id
            
        super().__init__(
            message=message,
            error_code='MODEL_NOT_FOUND',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ModelManagerError(AIServiceError):
    """Exception for ML model management errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='MODEL_MANAGER_ERROR',
            service_name='ml_model_manager',
            **kwargs
        )


class ABTestingError(AIServiceError):
    """Exception for A/B testing framework errors."""
    
    def __init__(self, message: str, test_id: str = None, **kwargs):
        details = kwargs.get('details', {})
        if test_id:
            details['test_id'] = test_id
            
        super().__init__(
            message=message,
            error_code='AB_TESTING_ERROR',
            service_name='ab_testing_framework',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ExperimentTrackerError(AIServiceError):
    """Exception for experiment tracker errors."""
    
    def __init__(self, message: str, experiment_id: str = None, **kwargs):
        details = kwargs.get('details', {})
        if experiment_id:
            details['experiment_id'] = experiment_id
            
        super().__init__(
            message=message,
            error_code='EXPERIMENT_TRACKER_ERROR',
            service_name='experiment_tracker',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class RLAgentError(AIServiceError):
    """Exception for reinforcement learning agent errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='RL_AGENT_ERROR',
            service_name='reinforcement_learning_agent',
            **kwargs
        )


class OptimizerError(AIServiceError):
    """Exception for optimizer errors."""
    
    def __init__(self, message: str, optimizer_type: str = None, **kwargs):
        details = kwargs.get('details', {})
        if optimizer_type:
            details['optimizer_type'] = optimizer_type
            
        super().__init__(
            message=message,
            error_code='OPTIMIZER_ERROR',
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )