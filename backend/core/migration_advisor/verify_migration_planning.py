"""
Verification script for Migration Planning Engine

This script verifies that all components are properly implemented.
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal


def verify_imports():
    """Verify all components can be imported."""
    print("Verifying imports...")
    
    try:
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
        print("✓ All components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_dependency_analyzer():
    """Verify DependencyAnalyzer implementation."""
    print("\nVerifying DependencyAnalyzer...")
    
    try:
        from .migration_planning_engine import DependencyAnalyzer
        
        analyzer = DependencyAnalyzer(None)
        
        # Check methods exist
        assert hasattr(analyzer, 'build_dependency_graph')
        assert hasattr(analyzer, 'discover_dependencies')
        assert hasattr(analyzer, 'validate_dependencies')
        
        print("✓ DependencyAnalyzer has all required methods")
        
        # Test cycle detection
        nodes = [
            {'id': 'w1', 'name': 'W1', 'type': 'app', 'complexity': 5},
            {'id': 'w2', 'name': 'W2', 'type': 'app', 'complexity': 5}
        ]
        edges = [
            {'from': 'w1', 'to': 'w2', 'type': 'depends_on', 'strength': 'strong'},
            {'from': 'w2', 'to': 'w1', 'type': 'depends_on', 'strength': 'strong'}
        ]
        
        cycles = analyzer._detect_cycles(nodes, edges)
        assert len(cycles) > 0, "Should detect circular dependency"
        
        print("✓ DependencyAnalyzer cycle detection works")
        return True
        
    except Exception as e:
        print(f"✗ DependencyAnalyzer verification failed: {e}")
        return False


def verify_migration_sequencer():
    """Verify MigrationSequencer implementation."""
    print("\nVerifying MigrationSequencer...")
    
    try:
        from .migration_planning_engine import MigrationSequencer
        
        sequencer = MigrationSequencer(None)
        
        # Check methods exist
        assert hasattr(sequencer, 'generate_migration_sequence')
        assert hasattr(sequencer, 'optimize_sequence_for_downtime')
        assert hasattr(sequencer, 'validate_migration_prerequisites')
        
        print("✓ MigrationSequencer has all required methods")
        
        # Test wave generation
        nodes = [
            {'id': 'w1', 'name': 'DB', 'type': 'database', 'complexity': 3},
            {'id': 'w2', 'name': 'App', 'type': 'application', 'complexity': 5}
        ]
        edges = [
            {'from': 'w1', 'to': 'w2', 'type': 'depends_on', 'strength': 'strong'}
        ]
        
        waves = sequencer._generate_optimized_waves(nodes, edges, 2, [])
        assert len(waves) > 0, "Should generate migration waves"
        
        print("✓ MigrationSequencer wave generation works")
        return True
        
    except Exception as e:
        print(f"✗ MigrationSequencer verification failed: {e}")
        return False


def verify_cost_estimator():
    """Verify MigrationCostEstimator implementation."""
    print("\nVerifying MigrationCostEstimator...")
    
    try:
        from .migration_planning_engine import MigrationCostEstimator
        
        estimator = MigrationCostEstimator(None)
        
        # Check methods exist
        assert hasattr(estimator, 'estimate_migration_costs')
        assert hasattr(estimator, 'estimate_data_transfer_cost')
        assert hasattr(estimator, 'estimate_dual_running_cost')
        assert hasattr(estimator, 'estimate_phase_cost')
        
        print("✓ MigrationCostEstimator has all required methods")
        
        # Test cost calculation constants
        assert estimator.DATA_TRANSFER_COST_PER_GB > 0
        assert estimator.DUAL_RUNNING_MULTIPLIER > 0
        assert estimator.PROFESSIONAL_SERVICES_HOURLY > 0
        
        print("✓ MigrationCostEstimator constants defined")
        return True
        
    except Exception as e:
        print(f"✗ MigrationCostEstimator verification failed: {e}")
        return False


def verify_progress_tracker():
    """Verify MigrationProgressTracker implementation."""
    print("\nVerifying MigrationProgressTracker...")
    
    try:
        from .migration_planning_engine import MigrationProgressTracker
        
        tracker = MigrationProgressTracker(None)
        
        # Check methods exist
        assert hasattr(tracker, 'get_migration_status')
        assert hasattr(tracker, 'update_phase_status')
        assert hasattr(tracker, 'track_workload_migration')
        assert hasattr(tracker, 'calculate_progress')
        
        print("✓ MigrationProgressTracker has all required methods")
        return True
        
    except Exception as e:
        print(f"✗ MigrationProgressTracker verification failed: {e}")
        return False


def verify_validator():
    """Verify MigrationValidator implementation."""
    print("\nVerifying MigrationValidator...")
    
    try:
        from .migration_planning_engine import MigrationValidator
        
        validator = MigrationValidator(None)
        
        # Check methods exist
        assert hasattr(validator, 'validate_phase_completion')
        assert hasattr(validator, 'validate_resource_deployment')
        assert hasattr(validator, 'validate_connectivity')
        assert hasattr(validator, 'validate_functionality')
        assert hasattr(validator, 'generate_validation_report')
        
        print("✓ MigrationValidator has all required methods")
        return True
        
    except Exception as e:
        print(f"✗ MigrationValidator verification failed: {e}")
        return False


def verify_rollback_manager():
    """Verify RollbackManager implementation."""
    print("\nVerifying RollbackManager...")
    
    try:
        from .migration_planning_engine import RollbackManager
        
        manager = RollbackManager(None)
        
        # Check methods exist
        assert hasattr(manager, 'generate_rollback_plan')
        assert hasattr(manager, 'execute_rollback')
        assert hasattr(manager, 'validate_rollback_readiness')
        assert hasattr(manager, 'get_rollback_history')
        
        print("✓ RollbackManager has all required methods")
        return True
        
    except Exception as e:
        print(f"✗ RollbackManager verification failed: {e}")
        return False


def verify_main_engine():
    """Verify MigrationPlanningEngine integration."""
    print("\nVerifying MigrationPlanningEngine...")
    
    try:
        from .migration_planning_engine import MigrationPlanningEngine
        
        engine = MigrationPlanningEngine(None)
        
        # Check all sub-components are initialized
        assert hasattr(engine, 'plan_generator')
        assert hasattr(engine, 'dependency_analyzer')
        assert hasattr(engine, 'sequencer')
        assert hasattr(engine, 'cost_estimator')
        assert hasattr(engine, 'progress_tracker')
        assert hasattr(engine, 'validator')
        assert hasattr(engine, 'rollback_manager')
        
        print("✓ MigrationPlanningEngine has all sub-components")
        
        # Check main methods exist
        methods = [
            'generate_migration_plan',
            'analyze_dependencies',
            'generate_migration_sequence',
            'validate_phase_prerequisites',
            'estimate_migration_costs',
            'get_migration_progress',
            'update_phase_status',
            'track_workload_progress',
            'validate_phase',
            'validate_workload_deployment',
            'validate_workload_connectivity',
            'validate_workload_functionality',
            'generate_phase_validation_report',
            'generate_rollback_plan',
            'execute_rollback',
            'validate_rollback_readiness',
            'get_rollback_history'
        ]
        
        for method in methods:
            assert hasattr(engine, method), f"Missing method: {method}"
        
        print(f"✓ MigrationPlanningEngine has all {len(methods)} required methods")
        return True
        
    except Exception as e:
        print(f"✗ MigrationPlanningEngine verification failed: {e}")
        return False


def main():
    """Run all verifications."""
    print("=" * 60)
    print("Migration Planning Engine Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", verify_imports()))
    results.append(("DependencyAnalyzer", verify_dependency_analyzer()))
    results.append(("MigrationSequencer", verify_migration_sequencer()))
    results.append(("MigrationCostEstimator", verify_cost_estimator()))
    results.append(("MigrationProgressTracker", verify_progress_tracker()))
    results.append(("MigrationValidator", verify_validator()))
    results.append(("RollbackManager", verify_rollback_manager()))
    results.append(("MigrationPlanningEngine", verify_main_engine()))
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All verifications passed!")
        return 0
    else:
        print("✗ Some verifications failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
