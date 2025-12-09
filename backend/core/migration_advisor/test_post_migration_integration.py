"""
Tests for Post-Migration Integration Engine

This module tests the post-migration integration functionality including
cost tracking, FinOps integration, baseline capture, governance, optimization,
and report generation.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock
from sqlalchemy.orm import Session

from .post_migration_integration_engine import (
    CostTrackingIntegrator,
    FinOpsConnector,
    BaselineCaptureSystem,
    GovernancePolicyApplicator,
    OptimizationIdentifier,
    MigrationReportGenerator,
    PostMigrationIntegrationEngine
)
from .models import (
    MigrationProject,
    OrganizationalStructure,
    CategorizedResource,
    MigrationPlan,
    BaselineMetrics,
    MigrationReport,
    MigrationStatus,
    OwnershipStatus,
    MigrationRiskLevel
)


class TestCostTrackingIntegrator:
    """Test cost tracking integration functionality."""
    
    def test_configure_cost_tracking(self):
        """Test cost tracking configuration."""
        db = Mock(spec=Session)
        integrator = CostTrackingIntegrator(db)
        
        # Create test organizational structure
        org_structure = OrganizationalStructure(
            migration_project_id="test-project-id",
            structure_name="Test Structure",
            teams=[
                {"name": "Engineering", "id": "eng", "owner": "john@example.com"},
                {"name": "Operations", "id": "ops", "owner": "jane@example.com"}
            ],
            projects=[
                {"name": "Project A", "cost_center": "CC-ENG"},
                {"name": "Project B", "cost_center": "CC-OPS"}
            ],
            cost_centers=[
                {"name": "Engineering CC", "code": "CC-ENG", "owner": "john@example.com"},
                {"name": "Operations CC", "code": "CC-OPS", "owner": "jane@example.com"}
            ]
        )
        
        result = integrator.configure_cost_tracking("test-project", org_structure)
        
        assert "cost_centers_created" in result
        assert "attribution_rules" in result
        assert "cost_center_mappings" in result
        assert len(result["cost_centers_created"]) == 2
        assert len(result["attribution_rules"]) > 0


class TestFinOpsConnector:
    """Test FinOps platform connector functionality."""
    
    def test_transfer_organizational_structure(self):
        """Test organizational structure transfer."""
        db = Mock(spec=Session)
        connector = FinOpsConnector(db)
        
        org_structure = OrganizationalStructure(
            migration_project_id="test-project-id",
            structure_name="Test Structure",
            teams=[{"name": "Engineering", "owner": "john@example.com"}],
            projects=[{"name": "Project A", "team": "Engineering"}],
            cost_centers=[{"name": "Engineering CC", "code": "CC-ENG"}]
        )
        
        result = connector.transfer_organizational_structure("test-project", org_structure)
        
        assert "teams_created" in result
        assert "projects_created" in result
        assert "cost_centers_created" in result
        assert len(result["teams_created"]) == 1
        assert len(result["projects_created"]) == 1
        assert len(result["cost_centers_created"]) == 1
    
    def test_enable_finops_features(self):
        """Test FinOps feature enablement."""
        db = Mock(spec=Session)
        connector = FinOpsConnector(db)
        
        result = connector.enable_finops_features("test-project", "AWS")
        
        assert "features_enabled" in result
        assert len(result["features_enabled"]) > 0
        
        # Check that key features are enabled
        feature_names = [f["feature"] for f in result["features_enabled"]]
        assert "waste_detection" in feature_names
        assert "ri_optimization" in feature_names
        assert "cost_anomaly_detection" in feature_names


class TestBaselineCaptureSystem:
    """Test baseline capture functionality."""
    
    def test_capture_baselines(self):
        """Test baseline metrics capture."""
        db = Mock(spec=Session)
        db.query = Mock()
        db.add = Mock()
        db.commit = Mock()
        db.refresh = Mock()
        
        # Mock project query
        mock_project = MagicMock()
        mock_project.id = "project-uuid"
        db.query.return_value.filter.return_value.first.return_value = mock_project
        
        capture_system = BaselineCaptureSystem(db)
        
        # Create test resources
        resources = [
            CategorizedResource(
                resource_id="resource-1",
                resource_type="compute_instance",
                team="Engineering",
                project="Project A",
                environment="production"
            ),
            CategorizedResource(
                resource_id="resource-2",
                resource_type="storage_bucket",
                team="Engineering",
                project="Project A",
                environment="production"
            )
        ]
        
        baseline = capture_system.capture_baselines("test-project", resources)
        
        assert baseline.resource_count == 2
        assert baseline.total_monthly_cost > 0
        assert "compute" in baseline.cost_by_service or "storage" in baseline.cost_by_service


class TestGovernancePolicyApplicator:
    """Test governance policy application."""
    
    def test_apply_tagging_policies(self):
        """Test tagging policy application."""
        db = Mock(spec=Session)
        applicator = GovernancePolicyApplicator(db)
        
        # Create test resources with missing tags
        resources = [
            CategorizedResource(
                resource_id="resource-1",
                resource_type="compute_instance",
                team="Engineering",
                project="Project A",
                environment="production",
                tags={}
            ),
            CategorizedResource(
                resource_id="resource-2",
                resource_type="storage_bucket",
                team="Engineering",
                project="Project A",
                environment="production",
                tags={"team": "Engineering"}
            )
        ]
        
        result = applicator.apply_tagging_policies("test-project", resources)
        
        assert "resources_processed" in result
        assert "resources_compliant" in result
        assert "resources_non_compliant" in result
        assert result["resources_processed"] == 2


class TestOptimizationIdentifier:
    """Test optimization identification."""
    
    def test_identify_optimization_opportunities(self):
        """Test optimization opportunity identification."""
        db = Mock(spec=Session)
        identifier = OptimizationIdentifier(db)
        
        # Create test baseline with underutilized resources
        baseline = BaselineMetrics(
            migration_project_id="project-uuid",
            total_monthly_cost=Decimal("1000.00"),
            resource_count=5,
            resource_utilization={
                "resource-1": {"cpu_utilization": 5.0, "memory_utilization": 8.0},
                "resource-2": {"cpu_utilization": 45.0, "memory_utilization": 60.0}
            }
        )
        
        resources = [
            CategorizedResource(
                resource_id="resource-1",
                resource_type="compute_instance"
            ),
            CategorizedResource(
                resource_id="resource-2",
                resource_type="compute_instance"
            )
        ]
        
        result = identifier.identify_optimization_opportunities(
            "test-project",
            baseline,
            resources
        )
        
        assert "opportunities" in result
        assert "total_potential_savings" in result
        assert len(result["opportunities"]) > 0


class TestMigrationReportGenerator:
    """Test migration report generation."""
    
    def test_generate_migration_report(self):
        """Test migration report generation."""
        db = Mock(spec=Session)
        db.query = Mock()
        db.add = Mock()
        db.commit = Mock()
        db.refresh = Mock()
        
        # Mock project query
        mock_project = MagicMock()
        mock_project.id = "project-uuid"
        mock_project.project_id = "test-project"
        mock_project.created_at = datetime(2024, 1, 1)
        mock_project.actual_completion = datetime(2024, 2, 1)
        mock_project.status = MigrationStatus.COMPLETE
        
        # Mock migration plan query
        mock_plan = MagicMock()
        mock_plan.total_duration_days = 30
        mock_plan.estimated_cost = Decimal("5000.00")
        
        # Mock baseline query
        mock_baseline = MagicMock()
        mock_baseline.total_monthly_cost = Decimal("4500.00")
        mock_baseline.cost_by_service = {"compute": 2000.0, "storage": 1000.0}
        mock_baseline.resource_utilization = {}
        
        # Setup query chain
        query_mock = Mock()
        filter_mock = Mock()
        order_by_mock = Mock()
        
        query_mock.filter.return_value = filter_mock
        filter_mock.first.side_effect = [mock_project, mock_plan, mock_baseline]
        filter_mock.all.return_value = []
        filter_mock.order_by.return_value = order_by_mock
        order_by_mock.first.return_value = mock_baseline
        
        db.query.return_value = query_mock
        
        generator = MigrationReportGenerator(db)
        
        report = generator.generate_migration_report("test-project")
        
        assert report is not None
        assert report.resources_migrated == 0
        assert report.success_rate >= 0


class TestPostMigrationIntegrationEngine:
    """Test complete post-migration integration engine."""
    
    def test_integration_engine_initialization(self):
        """Test integration engine initialization."""
        db = Mock(spec=Session)
        engine = PostMigrationIntegrationEngine(db)
        
        assert engine.cost_tracking_integrator is not None
        assert engine.finops_connector is not None
        assert engine.baseline_capture is not None
        assert engine.governance_applicator is not None
        assert engine.optimization_identifier is not None
        assert engine.report_generator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
