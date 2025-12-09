"""
Tests for Migration Planning Engine

This module contains tests for the migration planning engine components.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session

from .migration_planning_engine import (
    MigrationPlanGenerator,
    DependencyAnalyzer,
    MigrationSequencer,
    MigrationCostEstimator,
    MigrationProgressTracker,
    MigrationValidator,
    RollbackManager,
    MigrationPlanningEngine
)
from .models import (
    MigrationProject,
    OrganizationProfile,
    WorkloadProfile,
    MigrationPlan,
    MigrationPhase,
    MigrationStatus,
    PhaseStatus,
    CompanySize,
    InfrastructureType,
    ExperienceLevel,
    MigrationRiskLevel
)


def test_dependency_analyzer_builds_graph(db_session: Session):
    """Test that dependency analyzer can build a dependency graph."""
    analyzer = DependencyAnalyzer(db_session)
    
    # Create test workload profiles
    workload1 = WorkloadProfile(
        migration_project_id='test-project-id',
        workload_name='Database',
        application_type='database',
        total_compute_cores=4,
        total_memory_gb=16,
        total_storage_tb=1.0,
        data_volume_tb=0.5,
        dependencies=[]
    )
    
    workload2 = WorkloadProfile(
        migration_project_id='test-project-id',
        workload_name='Application',
        application_type='application',
        total_compute_cores=8,
        total_memory_gb=32,
        total_storage_tb=0.5,
        data_volume_tb=0.1,
        dependencies=[str(workload1.id)]
    )
    
    workloads = [workload1, workload2]
    
    # Build dependency graph
    graph = analyzer.build_dependency_graph(workloads)
    
    # Verify graph structure
    assert 'nodes' in graph
    assert 'edges' in graph
    assert 'critical_path' in graph
    assert 'migration_waves' in graph
    assert len(graph['nodes']) == 2
    assert len(graph['edges']) == 1
    
    # Verify edge connects workload2 to workload1
    edge = graph['edges'][0]
    assert edge['from'] == str(workload1.id)
    assert edge['to'] == str(workload2.id)


def test_dependency_analyzer_detects_cycles():
    """Test that dependency analyzer can detect circular dependencies."""
    analyzer = DependencyAnalyzer(None)
    
    # Create a circular dependency
    nodes = [
        {'id': 'w1', 'name': 'Workload 1', 'type': 'app', 'complexity': 5},
        {'id': 'w2', 'name': 'Workload 2', 'type': 'app', 'complexity': 5},
        {'id': 'w3', 'name': 'Workload 3', 'type': 'app', 'complexity': 5}
    ]
    
    edges = [
        {'from': 'w1', 'to': 'w2', 'type': 'depends_on', 'strength': 'strong'},
        {'from': 'w2', 'to': 'w3', 'type': 'depends_on', 'strength': 'strong'},
        {'from': 'w3', 'to': 'w1', 'type': 'depends_on', 'strength': 'strong'}  # Creates cycle
    ]
    
    graph = {'nodes': nodes, 'edges': edges}
    validation = analyzer.validate_dependencies(graph)
    
    assert validation['has_cycles'] is True
    assert len(validation['cycles']) > 0
    assert validation['is_valid'] is False


def test_migration_sequencer_generates_waves():
    """Test that migration sequencer generates migration waves."""
    sequencer = MigrationSequencer(None)
    
    # Create dependency graph
    nodes = [
        {'id': 'w1', 'name': 'Database', 'type': 'database', 'complexity': 3},
        {'id': 'w2', 'name': 'App Server', 'type': 'application', 'complexity': 5},
        {'id': 'w3', 'name': 'Web Server', 'type': 'web', 'complexity': 2}
    ]
    
    edges = [
        {'from': 'w1', 'to': 'w2', 'type': 'depends_on', 'strength': 'strong'}
    ]
    
    dependency_graph = {
        'nodes': nodes,
        'edges': edges
    }
    
    # Generate sequence
    sequence = sequencer.generate_migration_sequence(
        dependency_graph,
        [],
        {'max_parallel_migrations': 2}
    )
    
    assert 'waves' in sequence
    assert len(sequence['waves']) > 0
    assert 'total_downtime_hours' in sequence
    
    # First wave should contain w1 (no dependencies)
    first_wave = sequence['waves'][0]
    assert 'w1' in first_wave['workload_ids']


def test_cost_estimator_calculates_costs():
    """Test that cost estimator calculates migration costs."""
    estimator = MigrationCostEstimator(None)
    
    # Create test workload
    workload = WorkloadProfile(
        migration_project_id='test-project-id',
        workload_name='Test Workload',
        application_type='application',
        total_compute_cores=8,
        total_memory_gb=32,
        total_storage_tb=2.0,
        data_volume_tb=1.5
    )
    
    # Estimate data transfer cost
    transfer_cost = estimator._estimate_data_transfer_cost([workload])
    
    assert transfer_cost > 0
    assert isinstance(transfer_cost, Decimal)
    
    # Estimate dual-running cost
    dual_cost = estimator._estimate_dual_running_cost([workload], 30)
    
    assert dual_cost > 0
    assert isinstance(dual_cost, Decimal)


def test_progress_tracker_calculates_progress():
    """Test that progress tracker calculates migration progress."""
    tracker = MigrationProgressTracker(None)
    
    # Create test phases
    now = datetime.utcnow()
    phases = [
        MigrationPhase(
            phase_id='phase-1',
            migration_plan_id='plan-id',
            phase_name='Phase 1',
            phase_order=1,
            workloads=['w1', 'w2'],
            start_date=now - timedelta(days=10),
            end_date=now - timedelta(days=5),
            status=PhaseStatus.COMPLETED
        ),
        MigrationPhase(
            phase_id='phase-2',
            migration_plan_id='plan-id',
            phase_name='Phase 2',
            phase_order=2,
            workloads=['w3', 'w4'],
            start_date=now - timedelta(days=4),
            end_date=now + timedelta(days=2),
            status=PhaseStatus.IN_PROGRESS
        ),
        MigrationPhase(
            phase_id='phase-3',
            migration_plan_id='plan-id',
            phase_name='Phase 3',
            phase_order=3,
            workloads=['w5'],
            start_date=now + timedelta(days=3),
            end_date=now + timedelta(days=8),
            status=PhaseStatus.NOT_STARTED
        )
    ]
    
    # Calculate progress
    progress = tracker._calculate_overall_progress(phases)
    
    assert 0 <= progress <= 100
    assert progress > 0  # Should have some progress since one phase is complete


def test_validator_validates_phase():
    """Test that validator can validate phase completion."""
    validator = MigrationValidator(None)
    
    # Create test phase
    phase = MigrationPhase(
        phase_id='test-phase',
        migration_plan_id='plan-id',
        phase_name='Test Phase',
        phase_order=1,
        workloads=['w1'],
        status=PhaseStatus.COMPLETED,
        success_criteria=['All workloads migrated', 'No data loss']
    )
    
    # Validate criterion
    result = validator._validate_criterion('All workloads migrated', phase)
    
    # Completed phases should pass validation
    assert result is True


def test_rollback_manager_generates_plan():
    """Test that rollback manager generates rollback plans."""
    manager = RollbackManager(None)
    
    # Create test phase
    phase = MigrationPhase(
        phase_id='test-phase',
        migration_plan_id='plan-id',
        phase_name='Test Phase',
        phase_order=1,
        workloads=['w1', 'w2'],
        status=PhaseStatus.IN_PROGRESS
    )
    
    # Create test migration plan
    migration_plan = MigrationPlan(
        plan_id='test-plan',
        migration_project_id='project-id',
        target_provider='AWS',
        total_duration_days=30,
        estimated_cost=Decimal('50000.00'),
        risk_level=MigrationRiskLevel.MEDIUM
    )
    
    # Generate rollback steps
    steps = manager._generate_rollback_steps(phase, migration_plan)
    
    assert len(steps) > 0
    assert all('step_number' in step for step in steps)
    assert all('step_name' in step for step in steps)
    assert all('estimated_duration_minutes' in step for step in steps)


def test_rollback_manager_assesses_risk():
    """Test that rollback manager assesses rollback risk."""
    manager = RollbackManager(None)
    
    # Create test phase with many workloads
    phase = MigrationPhase(
        phase_id='test-phase',
        migration_plan_id='plan-id',
        phase_name='Test Phase',
        phase_order=1,
        workloads=[f'w{i}' for i in range(15)],  # 15 workloads
        status=PhaseStatus.IN_PROGRESS
    )
    
    # Create low-risk migration plan
    migration_plan = MigrationPlan(
        plan_id='test-plan',
        migration_project_id='project-id',
        target_provider='AWS',
        total_duration_days=30,
        estimated_cost=Decimal('50000.00'),
        risk_level=MigrationRiskLevel.LOW
    )
    
    # Assess risk
    risk = manager._assess_rollback_risk(phase, migration_plan)
    
    # Risk should be elevated due to many workloads
    assert risk in ['low', 'medium', 'high', 'critical']


if __name__ == '__main__':
    print("Migration Planning Engine tests")
    print("Run with: pytest test_migration_planning_engine.py")
