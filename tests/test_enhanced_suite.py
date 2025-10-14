"""
Comprehensive test suite for enhanced data models and schedulers
Runs all tests for task 1.3: Write unit tests for enhanced data models and schedulers
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from tests.test_enhanced_models import *
from tests.test_enhanced_workload_vm import *
from tests.test_enhanced_schedulers_fixed import *


def run_enhanced_tests():
    """Run all enhanced model and scheduler tests"""
    print("=" * 60)
    print("ENHANCED DATA MODELS AND SCHEDULERS TEST SUITE")
    print("=" * 60)
    print("Task 1.3: Write unit tests for enhanced data models and schedulers")
    print("- Create unit tests for new data model validation and serialization")
    print("- Test scheduler decision-making logic with various constraint combinations")
    print("- Validate cost calculation and performance prediction accuracy")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    print("\n📋 Loading test modules...")
    
    # Enhanced models tests
    suite.addTests(loader.loadTestsFromTestCase(TestCostConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceRequirements))
    suite.addTests(loader.loadTestsFromTestCase(TestComplianceRequirements))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    print("✅ Enhanced data models tests loaded")
    
    # Enhanced workload and VM tests
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedWorkload))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedVirtualMachine))
    print("✅ Enhanced workload and VM tests loaded")
    
    # Enhanced scheduler tests
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestCostAwareScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulerIntegration))
    print("✅ Enhanced scheduler tests loaded")
    
    # Run tests
    print("\n🧪 Running tests...")
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Task 1.3 Requirements Validated:")
        print("  ✓ Data model validation and serialization tests")
        print("  ✓ Scheduler decision-making logic tests")
        print("  ✓ Cost calculation and performance prediction tests")
        print("  ✓ Various constraint combination tests")
        print("  ✓ Enhanced workload and VM functionality tests")
        return True
    else:
        print("\n❌ Some tests failed. Please review the failures above.")
        return False


if __name__ == '__main__':
    success = run_enhanced_tests()
    sys.exit(0 if success else 1)