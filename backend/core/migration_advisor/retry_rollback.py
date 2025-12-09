"""
Retry and Rollback Mechanisms for Migration Advisor

This module provides exponential backoff retry logic for API calls,
rollback procedures for failed operations, and transaction management
for critical operations.
"""

import time
import logging
from typing import Callable, Any, Optional, Dict, List, TypeVar, Generic
from datetime import datetime
from functools import wraps
from enum import Enum

from ..exceptions import (
    CloudProviderException,
    CloudProviderRateLimitException,
    DatabaseException,
)
from .models import MigrationPhase, PhaseStatus


logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions: tuple = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.strategy = strategy
        self.retryable_exceptions = retryable_exceptions or (
            CloudProviderException,
            CloudProviderRateLimitException,
            DatabaseException,
        )
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.initial_delay * attempt
        else:  # FIXED_DELAY
            delay = self.initial_delay
        
        return min(delay, self.max_delay)


class RetryResult(Generic[T]):
    """Result of a retry operation"""
    
    def __init__(
        self,
        success: bool,
        result: T = None,
        attempts: int = 0,
        total_delay: float = 0.0,
        last_exception: Exception = None
    ):
        self.success = success
        self.result = result
        self.attempts = attempts
        self.total_delay = total_delay
        self.last_exception = last_exception


class RetryManager:
    """
    Manager for retry operations with exponential backoff
    
    Provides configurable retry logic for API calls and other operations
    that may fail transiently.
    """
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation",
        context: Dict[str, Any] = None
    ) -> RetryResult[T]:
        """
        Execute operation with retry logic
        
        Args:
            operation: Callable to execute
            operation_name: Name for logging
            context: Additional context for logging
            
        Returns:
            RetryResult with operation result or exception
        """
        context = context or {}
        attempts = 0
        total_delay = 0.0
        last_exception = None
        
        while attempts < self.config.max_attempts:
            attempts += 1
            
            try:
                logger.info(
                    f"Executing {operation_name} (attempt {attempts}/{self.config.max_attempts})",
                    extra=context
                )
                
                result = operation()
                
                logger.info(
                    f"{operation_name} succeeded on attempt {attempts}",
                    extra=context
                )
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_delay=total_delay
                )
            
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                # Check if this is a rate limit exception with retry_after
                if isinstance(e, CloudProviderRateLimitException):
                    retry_after = getattr(e, 'details', {}).get('retry_after', None)
                    if retry_after:
                        delay = retry_after
                    else:
                        delay = self.config.calculate_delay(attempts)
                else:
                    delay = self.config.calculate_delay(attempts)
                
                logger.warning(
                    f"{operation_name} failed on attempt {attempts}: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds...",
                    extra={**context, 'exception': str(e), 'delay': delay}
                )
                
                if attempts < self.config.max_attempts:
                    time.sleep(delay)
                    total_delay += delay
            
            except Exception as e:
                # Non-retryable exception
                logger.error(
                    f"{operation_name} failed with non-retryable exception: {str(e)}",
                    extra={**context, 'exception': str(e)}
                )
                
                return RetryResult(
                    success=False,
                    attempts=attempts,
                    total_delay=total_delay,
                    last_exception=e
                )
        
        # Max attempts reached
        logger.error(
            f"{operation_name} failed after {attempts} attempts",
            extra={**context, 'last_exception': str(last_exception)}
        )
        
        return RetryResult(
            success=False,
            attempts=attempts,
            total_delay=total_delay,
            last_exception=last_exception
        )


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
):
    """
    Decorator for automatic retry with exponential backoff
    
    Usage:
        @with_retry(max_attempts=5, initial_delay=2.0)
        def my_api_call():
            # API call logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                strategy=strategy
            )
            
            manager = RetryManager(config)
            
            def operation():
                return func(*args, **kwargs)
            
            result = manager.execute_with_retry(
                operation=operation,
                operation_name=func.__name__
            )
            
            if result.success:
                return result.result
            else:
                raise result.last_exception
        
        return wrapper
    return decorator


class RollbackAction:
    """Represents a single rollback action"""
    
    def __init__(
        self,
        action_id: str,
        description: str,
        rollback_func: Callable,
        context: Dict[str, Any] = None
    ):
        self.action_id = action_id
        self.description = description
        self.rollback_func = rollback_func
        self.context = context or {}
        self.executed = False
        self.execution_time: Optional[datetime] = None
        self.success: Optional[bool] = None
        self.error: Optional[Exception] = None
    
    def execute(self) -> bool:
        """Execute the rollback action"""
        try:
            logger.info(f"Executing rollback action: {self.description}")
            self.rollback_func(**self.context)
            self.executed = True
            self.execution_time = datetime.utcnow()
            self.success = True
            logger.info(f"Rollback action completed: {self.description}")
            return True
        
        except Exception as e:
            logger.error(
                f"Rollback action failed: {self.description}",
                extra={'error': str(e)}
            )
            self.executed = True
            self.execution_time = datetime.utcnow()
            self.success = False
            self.error = e
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action_id': self.action_id,
            'description': self.description,
            'executed': self.executed,
            'execution_time': self.execution_time.isoformat() + 'Z' if self.execution_time else None,
            'success': self.success,
            'error': str(self.error) if self.error else None
        }


class RollbackManager:
    """
    Manager for rollback operations
    
    Maintains a stack of rollback actions and executes them in reverse order
    when a rollback is triggered.
    """
    
    def __init__(self):
        self.actions: List[RollbackAction] = []
        self.rollback_executed = False
        self.rollback_start_time: Optional[datetime] = None
        self.rollback_end_time: Optional[datetime] = None
    
    def register_action(
        self,
        action_id: str,
        description: str,
        rollback_func: Callable,
        context: Dict[str, Any] = None
    ):
        """
        Register a rollback action
        
        Args:
            action_id: Unique identifier for the action
            description: Human-readable description
            rollback_func: Function to execute for rollback
            context: Context data to pass to rollback function
        """
        action = RollbackAction(action_id, description, rollback_func, context)
        self.actions.append(action)
        logger.debug(f"Registered rollback action: {description}")
    
    def execute_rollback(self) -> Dict[str, Any]:
        """
        Execute all rollback actions in reverse order
        
        Returns:
            Dictionary with rollback results
        """
        if self.rollback_executed:
            logger.warning("Rollback already executed")
            return self.get_rollback_status()
        
        logger.info(f"Starting rollback of {len(self.actions)} actions")
        self.rollback_start_time = datetime.utcnow()
        self.rollback_executed = True
        
        # Execute in reverse order (LIFO)
        successful = 0
        failed = 0
        
        for action in reversed(self.actions):
            if action.execute():
                successful += 1
            else:
                failed += 1
        
        self.rollback_end_time = datetime.utcnow()
        
        logger.info(
            f"Rollback completed: {successful} successful, {failed} failed"
        )
        
        return self.get_rollback_status()
    
    def get_rollback_status(self) -> Dict[str, Any]:
        """Get current rollback status"""
        return {
            'rollback_executed': self.rollback_executed,
            'total_actions': len(self.actions),
            'executed_actions': sum(1 for a in self.actions if a.executed),
            'successful_actions': sum(1 for a in self.actions if a.success),
            'failed_actions': sum(1 for a in self.actions if a.executed and not a.success),
            'start_time': self.rollback_start_time.isoformat() + 'Z' if self.rollback_start_time else None,
            'end_time': self.rollback_end_time.isoformat() + 'Z' if self.rollback_end_time else None,
            'actions': [a.to_dict() for a in self.actions]
        }
    
    def clear(self):
        """Clear all rollback actions"""
        self.actions.clear()
        self.rollback_executed = False
        self.rollback_start_time = None
        self.rollback_end_time = None


class TransactionManager:
    """
    Transaction manager for critical operations
    
    Provides transaction-like semantics with automatic rollback on failure.
    """
    
    def __init__(self):
        self.rollback_manager = RollbackManager()
        self.in_transaction = False
        self.transaction_id: Optional[str] = None
        self.transaction_start_time: Optional[datetime] = None
    
    def begin_transaction(self, transaction_id: str):
        """Begin a new transaction"""
        if self.in_transaction:
            raise RuntimeError("Transaction already in progress")
        
        self.transaction_id = transaction_id
        self.transaction_start_time = datetime.utcnow()
        self.in_transaction = True
        self.rollback_manager.clear()
        
        logger.info(f"Transaction started: {transaction_id}")
    
    def register_rollback(
        self,
        action_id: str,
        description: str,
        rollback_func: Callable,
        context: Dict[str, Any] = None
    ):
        """Register a rollback action for current transaction"""
        if not self.in_transaction:
            raise RuntimeError("No transaction in progress")
        
        self.rollback_manager.register_action(
            action_id=action_id,
            description=description,
            rollback_func=rollback_func,
            context=context
        )
    
    def commit(self) -> Dict[str, Any]:
        """Commit the transaction"""
        if not self.in_transaction:
            raise RuntimeError("No transaction in progress")
        
        logger.info(f"Transaction committed: {self.transaction_id}")
        
        result = {
            'transaction_id': self.transaction_id,
            'status': 'committed',
            'start_time': self.transaction_start_time.isoformat() + 'Z',
            'end_time': datetime.utcnow().isoformat() + 'Z',
            'rollback_actions_registered': len(self.rollback_manager.actions)
        }
        
        self.in_transaction = False
        self.transaction_id = None
        self.transaction_start_time = None
        
        return result
    
    def rollback(self) -> Dict[str, Any]:
        """Rollback the transaction"""
        if not self.in_transaction:
            raise RuntimeError("No transaction in progress")
        
        logger.warning(f"Transaction rolling back: {self.transaction_id}")
        
        rollback_status = self.rollback_manager.execute_rollback()
        
        result = {
            'transaction_id': self.transaction_id,
            'status': 'rolled_back',
            'start_time': self.transaction_start_time.isoformat() + 'Z',
            'end_time': datetime.utcnow().isoformat() + 'Z',
            'rollback_status': rollback_status
        }
        
        self.in_transaction = False
        self.transaction_id = None
        self.transaction_start_time = None
        
        return result
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic rollback on exception"""
        if exc_type is not None:
            # Exception occurred, rollback
            if self.in_transaction:
                self.rollback()
            return False  # Re-raise exception
        else:
            # No exception, commit
            if self.in_transaction:
                self.commit()
            return True


class MigrationPhaseRollback:
    """
    Specialized rollback handler for migration phases
    
    Provides rollback procedures specific to migration operations.
    """
    
    def __init__(self, db_session):
        self.db_session = db_session
        self.transaction_manager = TransactionManager()
    
    def execute_phase_with_rollback(
        self,
        phase: MigrationPhase,
        operation: Callable,
        rollback_procedures: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute migration phase with automatic rollback on failure
        
        Args:
            phase: MigrationPhase to execute
            operation: Operation to perform
            rollback_procedures: List of rollback procedures
            
        Returns:
            Dictionary with execution results
        """
        transaction_id = f"phase-{phase.phase_id}-{int(time.time())}"
        
        try:
            self.transaction_manager.begin_transaction(transaction_id)
            
            # Register rollback procedures
            for i, procedure in enumerate(rollback_procedures):
                self.transaction_manager.register_rollback(
                    action_id=f"{transaction_id}-action-{i}",
                    description=procedure.get('description', f'Rollback action {i}'),
                    rollback_func=procedure['rollback_func'],
                    context=procedure.get('context', {})
                )
            
            # Update phase status to in progress
            phase.status = PhaseStatus.IN_PROGRESS
            phase.actual_start_date = datetime.utcnow()
            self.db_session.commit()
            
            # Execute operation
            result = operation()
            
            # Update phase status to completed
            phase.status = PhaseStatus.COMPLETED
            phase.actual_end_date = datetime.utcnow()
            self.db_session.commit()
            
            # Commit transaction
            transaction_result = self.transaction_manager.commit()
            
            return {
                'success': True,
                'phase_id': phase.phase_id,
                'result': result,
                'transaction': transaction_result
            }
        
        except Exception as e:
            logger.error(
                f"Phase execution failed: {phase.phase_id}",
                extra={'error': str(e)}
            )
            
            # Update phase status to failed
            phase.status = PhaseStatus.FAILED
            self.db_session.commit()
            
            # Rollback transaction
            rollback_result = self.transaction_manager.rollback()
            
            # Update phase status to rolled back if rollback successful
            if rollback_result['rollback_status']['successful_actions'] > 0:
                phase.status = PhaseStatus.ROLLED_BACK
                self.db_session.commit()
            
            return {
                'success': False,
                'phase_id': phase.phase_id,
                'error': str(e),
                'transaction': rollback_result
            }


# Global instances
_retry_manager = RetryManager()
_transaction_manager = TransactionManager()


def get_retry_manager() -> RetryManager:
    """Get the global retry manager instance"""
    return _retry_manager


def get_transaction_manager() -> TransactionManager:
    """Get the global transaction manager instance"""
    return _transaction_manager
