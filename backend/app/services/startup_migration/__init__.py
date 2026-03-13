"""
Startup Migration Module

A simplified migration module for startups to migrate databases from on-premises
to cloud (AWS, GCP, Azure) with cost comparison and migration planning.
"""

from backend.app.services.startup_migration.models import (
    StartupMigrationProject,
    DatabaseAssessment,
    CloudRecommendation,
    MigrationPlan,
    FinOpsIntegration
)

from .assessment_service import AssessmentService
from .pricing_service import MultiCloudPricingService
from backend.app.services.migration_advisor.migration_advisor.recommendation_engine import RecommendationEngine
from .migration_planner import MigrationPlanner
from .finops_integrator import FinOpsIntegrator

__all__ = [
    'StartupMigrationProject',
    'DatabaseAssessment',
    'CloudRecommendation',
    'MigrationPlan',
    'FinOpsIntegration',
    'AssessmentService',
    'MultiCloudPricingService',
    'RecommendationEngine',
    'MigrationPlanner',
    'FinOpsIntegrator',
]
