"""
Cloud Migration Advisor Module

This module provides intelligent guidance for cloud migration including:
- Migration assessment and requirements gathering
- Cloud provider recommendations
- Migration planning and execution support
- Resource organization and categorization
"""

from .models import (
    MigrationProject,
    OrganizationProfile,
    WorkloadProfile,
    PerformanceRequirements,
    ComplianceRequirements,
    BudgetConstraints,
    TechnicalRequirements,
    ProviderEvaluation,
    RecommendationReport,
    MigrationPlan,
    MigrationPhase,
    CategorizedResource,
    OrganizationalStructure,
)

from .assessment_engine import (
    MigrationAssessmentEngine,
    MigrationProjectManager,
    OrganizationProfiler,
    AssessmentTimelineEstimator,
)

from .requirements_analysis_engine import (
    WorkloadAnalysisEngine,
    WorkloadProfiler,
    PerformanceAnalyzer,
    ComplianceAssessor,
    BudgetAnalyzer,
    TechnicalRequirementsMapper,
    RequirementsCompletenessValidator,
)

from .resource_organization_engine import (
    ResourceOrganizationEngine,
    OrganizationResult,
)

from .resource_discovery_engine import (
    ResourceDiscoveryEngine,
    CloudProvider,
    CloudResource,
    ResourceInventory,
    ProviderCredentials,
)

from .organizational_structure_manager import (
    OrganizationalStructureManager,
    Team,
    Project,
    Environment,
    Region,
    CostCenter,
    DimensionType,
)

from .auto_categorization_engine import (
    AutoCategorizationEngine,
    ResourceCategorization,
    CategorizedResources,
    CategorizationRule,
)

from .tagging_engine import (
    TaggingEngine,
    TaggingPolicy,
    TagConflictResolution,
    BulkTaggingResult,
)

from .hierarchy_builder import (
    HierarchyBuilder,
    HierarchyView,
    HierarchyNode,
)

from .ownership_resolver import (
    OwnershipResolver,
    OwnershipSuggestion,
    OwnershipResolutionResult,
)

from .error_handler import (
    MigrationErrorHandler,
    MigrationError,
    AssessmentError,
    RecommendationError,
    MigrationExecutionError,
    OrganizationError,
    ErrorResponse,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    get_error_handler,
)

from .validation import (
    MigrationValidator,
    ValidationResult,
    ValidationError,
    FieldValidator,
    OrganizationProfileValidator,
    WorkloadProfileValidator,
    PerformanceRequirementsValidator,
    BudgetConstraintsValidator,
    MigrationPlanValidator,
    OrganizationalStructureValidator,
    CategorizedResourceValidator,
    get_validator,
)

from .retry_rollback import (
    RetryManager,
    RetryConfig,
    RetryStrategy,
    RetryResult,
    RollbackManager,
    RollbackAction,
    TransactionManager,
    MigrationPhaseRollback,
    with_retry,
    get_retry_manager,
    get_transaction_manager,
)

__all__ = [
    "MigrationProject",
    "OrganizationProfile",
    "WorkloadProfile",
    "PerformanceRequirements",
    "ComplianceRequirements",
    "BudgetConstraints",
    "TechnicalRequirements",
    "ProviderEvaluation",
    "RecommendationReport",
    "MigrationPlan",
    "MigrationPhase",
    "CategorizedResource",
    "OrganizationalStructure",
    "MigrationAssessmentEngine",
    "MigrationProjectManager",
    "OrganizationProfiler",
    "AssessmentTimelineEstimator",
    "WorkloadAnalysisEngine",
    "WorkloadProfiler",
    "PerformanceAnalyzer",
    "ComplianceAssessor",
    "BudgetAnalyzer",
    "TechnicalRequirementsMapper",
    "RequirementsCompletenessValidator",
    "ResourceOrganizationEngine",
    "OrganizationResult",
    "ResourceDiscoveryEngine",
    "CloudProvider",
    "CloudResource",
    "ResourceInventory",
    "ProviderCredentials",
    "OrganizationalStructureManager",
    "Team",
    "Project",
    "Environment",
    "Region",
    "CostCenter",
    "DimensionType",
    "AutoCategorizationEngine",
    "ResourceCategorization",
    "CategorizedResources",
    "CategorizationRule",
    "TaggingEngine",
    "TaggingPolicy",
    "TagConflictResolution",
    "BulkTaggingResult",
    "HierarchyBuilder",
    "HierarchyView",
    "HierarchyNode",
    "OwnershipResolver",
    "OwnershipSuggestion",
    "OwnershipResolutionResult",
    "MigrationErrorHandler",
    "MigrationError",
    "AssessmentError",
    "RecommendationError",
    "MigrationExecutionError",
    "OrganizationError",
    "ErrorResponse",
    "ErrorCategory",
    "ErrorSeverity",
    "RecoveryStrategy",
    "get_error_handler",
    "MigrationValidator",
    "ValidationResult",
    "ValidationError",
    "FieldValidator",
    "OrganizationProfileValidator",
    "WorkloadProfileValidator",
    "PerformanceRequirementsValidator",
    "BudgetConstraintsValidator",
    "MigrationPlanValidator",
    "OrganizationalStructureValidator",
    "CategorizedResourceValidator",
    "get_validator",
    "RetryManager",
    "RetryConfig",
    "RetryStrategy",
    "RetryResult",
    "RollbackManager",
    "RollbackAction",
    "TransactionManager",
    "MigrationPhaseRollback",
    "with_retry",
    "get_retry_manager",
    "get_transaction_manager",
]
