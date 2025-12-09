"""
Migration Advisor Error Handler

This module provides comprehensive error handling for the Cloud Migration Advisor,
including error categorization, recovery strategies, and detailed logging.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

from ..exceptions import (
    FinOpsException,
    ValidationException,
    CloudProviderException,
    DatabaseException,
    ExternalServiceException,
    ResourceNotFoundException,
)
from .models import MigrationProject, MigrationPhase


logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of migration errors"""
    ASSESSMENT_ERROR = "assessment_error"
    RECOMMENDATION_ERROR = "recommendation_error"
    MIGRATION_EXECUTION_ERROR = "migration_execution_error"
    RESOURCE_ORGANIZATION_ERROR = "resource_organization_error"
    VALIDATION_ERROR = "validation_error"
    PROVIDER_API_ERROR = "provider_api_error"
    DATABASE_ERROR = "database_error"
    CONFIGURATION_ERROR = "configuration_error"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    ROLLBACK = "rollback"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"


class MigrationError:
    """
    Structured migration error with categorization and context
    """
    
    def __init__(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: str,
        error_code: str,
        details: Dict[str, Any] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.MANUAL_INTERVENTION,
        project_id: str = None,
        phase_id: str = None,
        resource_id: str = None,
        original_exception: Exception = None
    ):
        self.category = category
        self.severity = severity
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.recovery_strategy = recovery_strategy
        self.project_id = project_id
        self.phase_id = phase_id
        self.resource_id = resource_id
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow()
        self.error_id = f"ERR-{self.timestamp.strftime('%Y%m%d%H%M%S')}-{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and API responses"""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "recovery_strategy": self.recovery_strategy.value,
            "project_id": self.project_id,
            "phase_id": self.phase_id,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp.isoformat() + "Z",
            "original_exception": str(self.original_exception) if self.original_exception else None
        }


class AssessmentError(MigrationError):
    """Error during assessment phase"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            category=ErrorCategory.ASSESSMENT_ERROR,
            severity=kwargs.pop('severity', ErrorSeverity.MEDIUM),
            message=message,
            error_code=kwargs.pop('error_code', 'ASSESSMENT_ERROR'),
            recovery_strategy=kwargs.pop('recovery_strategy', RecoveryStrategy.MANUAL_INTERVENTION),
            **kwargs
        )


class RecommendationError(MigrationError):
    """Error during recommendation generation"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            category=ErrorCategory.RECOMMENDATION_ERROR,
            severity=kwargs.pop('severity', ErrorSeverity.HIGH),
            message=message,
            error_code=kwargs.pop('error_code', 'RECOMMENDATION_ERROR'),
            recovery_strategy=kwargs.pop('recovery_strategy', RecoveryStrategy.FALLBACK),
            **kwargs
        )


class MigrationExecutionError(MigrationError):
    """Error during migration execution"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            category=ErrorCategory.MIGRATION_EXECUTION_ERROR,
            severity=kwargs.pop('severity', ErrorSeverity.CRITICAL),
            message=message,
            error_code=kwargs.pop('error_code', 'MIGRATION_EXECUTION_ERROR'),
            recovery_strategy=kwargs.pop('recovery_strategy', RecoveryStrategy.ROLLBACK),
            **kwargs
        )


class OrganizationError(MigrationError):
    """Error during resource organization"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            category=ErrorCategory.RESOURCE_ORGANIZATION_ERROR,
            severity=kwargs.pop('severity', ErrorSeverity.MEDIUM),
            message=message,
            error_code=kwargs.pop('error_code', 'ORGANIZATION_ERROR'),
            recovery_strategy=kwargs.pop('recovery_strategy', RecoveryStrategy.MANUAL_INTERVENTION),
            **kwargs
        )


class ErrorResponse:
    """Response object for error handling"""
    
    def __init__(
        self,
        error: MigrationError,
        recovery_action: str = None,
        can_retry: bool = False,
        retry_after: int = None,
        user_action_required: bool = False,
        user_action_message: str = None
    ):
        self.error = error
        self.recovery_action = recovery_action
        self.can_retry = can_retry
        self.retry_after = retry_after
        self.user_action_required = user_action_required
        self.user_action_message = user_action_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "error": self.error.to_dict(),
            "recovery_action": self.recovery_action,
            "can_retry": self.can_retry,
            "retry_after": self.retry_after,
            "user_action_required": self.user_action_required,
            "user_action_message": self.user_action_message
        }


class MigrationErrorHandler:
    """
    Comprehensive error handler for Cloud Migration Advisor
    
    Provides error categorization, recovery strategies, and detailed logging
    for all migration-related errors.
    """
    
    def __init__(self):
        self.error_history: List[MigrationError] = []
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {
            ErrorCategory.ASSESSMENT_ERROR: self._handle_assessment_error,
            ErrorCategory.RECOMMENDATION_ERROR: self._handle_recommendation_error,
            ErrorCategory.MIGRATION_EXECUTION_ERROR: self._handle_migration_error,
            ErrorCategory.RESOURCE_ORGANIZATION_ERROR: self._handle_organization_error,
        }
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None
    ) -> ErrorResponse:
        """
        Handle any error and return appropriate response
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            ErrorResponse with recovery information
        """
        context = context or {}
        
        # Convert exception to MigrationError
        migration_error = self._categorize_error(error, context)
        
        # Log the error
        self._log_error(migration_error)
        
        # Store in history
        self.error_history.append(migration_error)
        
        # Get recovery handler
        handler = self.recovery_handlers.get(
            migration_error.category,
            self._handle_generic_error
        )
        
        # Execute recovery strategy
        response = handler(migration_error, context)
        
        return response
    
    def _categorize_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> MigrationError:
        """Categorize exception into MigrationError"""
        
        # If already a MigrationError, return it
        if isinstance(error, MigrationError):
            return error
        
        # Categorize based on exception type
        if isinstance(error, ValidationException):
            return AssessmentError(
                message=str(error),
                error_code="VALIDATION_ERROR",
                details=getattr(error, 'details', {}),
                original_exception=error,
                **context
            )
        
        elif isinstance(error, CloudProviderException):
            return MigrationExecutionError(
                message=str(error),
                error_code="PROVIDER_API_ERROR",
                severity=ErrorSeverity.CRITICAL,
                details=getattr(error, 'details', {}),
                original_exception=error,
                **context
            )
        
        elif isinstance(error, DatabaseException):
            return MigrationError(
                category=ErrorCategory.DATABASE_ERROR,
                severity=ErrorSeverity.HIGH,
                message=str(error),
                error_code="DATABASE_ERROR",
                details=getattr(error, 'details', {}),
                recovery_strategy=RecoveryStrategy.RETRY,
                original_exception=error,
                **context
            )
        
        elif isinstance(error, ResourceNotFoundException):
            return OrganizationError(
                message=str(error),
                error_code="RESOURCE_NOT_FOUND",
                severity=ErrorSeverity.MEDIUM,
                details=getattr(error, 'details', {}),
                original_exception=error,
                **context
            )
        
        else:
            # Generic error
            return MigrationError(
                category=ErrorCategory.MIGRATION_EXECUTION_ERROR,
                severity=ErrorSeverity.HIGH,
                message=str(error),
                error_code="UNKNOWN_ERROR",
                details={"exception_type": type(error).__name__},
                recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                original_exception=error,
                **context
            )
    
    def _log_error(self, error: MigrationError):
        """Log error with appropriate level"""
        log_data = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical migration error: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity migration error: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity migration error: {error.message}", extra=log_data)
        else:
            logger.info(f"Low severity migration error: {error.message}", extra=log_data)
    
    def _handle_assessment_error(
        self,
        error: MigrationError,
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """
        Handle errors during assessment phase
        
        Strategy: Save progress, allow resumption, request user input
        """
        return ErrorResponse(
            error=error,
            recovery_action="Assessment progress has been saved. Please review the error and provide corrected information.",
            can_retry=True,
            user_action_required=True,
            user_action_message="Please review and correct the assessment data, then retry."
        )
    
    def _handle_recommendation_error(
        self,
        error: MigrationError,
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """
        Handle errors during recommendation generation
        
        Strategy: Fallback to rule-based recommendations, request additional data
        """
        return ErrorResponse(
            error=error,
            recovery_action="Falling back to rule-based recommendation engine. Consider providing additional requirement details for more accurate recommendations.",
            can_retry=True,
            user_action_required=False,
            user_action_message="ML-based recommendations unavailable. Using rule-based fallback. You may provide additional data to improve accuracy."
        )
    
    def _handle_migration_error(
        self,
        error: MigrationError,
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """
        Handle errors during migration execution
        
        Strategy: Automatic retry with exponential backoff, rollback on critical failures
        """
        if error.severity == ErrorSeverity.CRITICAL:
            return ErrorResponse(
                error=error,
                recovery_action="Critical migration error detected. Initiating rollback procedure.",
                can_retry=False,
                user_action_required=True,
                user_action_message="Critical error during migration. Rollback has been initiated. Please review the error details and contact support if needed."
            )
        else:
            return ErrorResponse(
                error=error,
                recovery_action="Migration error detected. Will retry with exponential backoff.",
                can_retry=True,
                retry_after=60,  # Retry after 60 seconds
                user_action_required=False
            )
    
    def _handle_organization_error(
        self,
        error: MigrationError,
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """
        Handle errors during resource organization
        
        Strategy: Manual intervention workflow, conflict resolution UI
        """
        return ErrorResponse(
            error=error,
            recovery_action="Resource organization error detected. Manual intervention required.",
            can_retry=True,
            user_action_required=True,
            user_action_message="Unable to automatically organize resources. Please review the conflict and provide manual categorization."
        )
    
    def _handle_generic_error(
        self,
        error: MigrationError,
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """Handle generic errors"""
        return ErrorResponse(
            error=error,
            recovery_action="An unexpected error occurred. Please review the error details.",
            can_retry=False,
            user_action_required=True,
            user_action_message="An unexpected error occurred. Please contact support with the error ID."
        )
    
    def get_error_history(
        self,
        project_id: str = None,
        category: ErrorCategory = None,
        severity: ErrorSeverity = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get error history with optional filtering
        
        Args:
            project_id: Filter by project ID
            category: Filter by error category
            severity: Filter by severity level
            limit: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
        """
        filtered_errors = self.error_history
        
        if project_id:
            filtered_errors = [e for e in filtered_errors if e.project_id == project_id]
        
        if category:
            filtered_errors = [e for e in filtered_errors if e.category == category]
        
        if severity:
            filtered_errors = [e for e in filtered_errors if e.severity == severity]
        
        # Sort by timestamp descending
        filtered_errors.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        filtered_errors = filtered_errors[:limit]
        
        return [e.to_dict() for e in filtered_errors]
    
    def clear_error_history(self, project_id: str = None):
        """Clear error history, optionally for a specific project"""
        if project_id:
            self.error_history = [e for e in self.error_history if e.project_id != project_id]
        else:
            self.error_history.clear()


# Global error handler instance
_error_handler = MigrationErrorHandler()


def get_error_handler() -> MigrationErrorHandler:
    """Get the global error handler instance"""
    return _error_handler
