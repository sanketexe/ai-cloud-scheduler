#!/usr/bin/env python3
"""
Main test runner for AI Cloud Scheduler Backend Tests
Run with: python run_tests.py
"""

import sys
import os
import argparse

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

def import_test_modules():
    """Import all test modules with error handling"""
    test_modules = {}
    
    try:
        from test_utils import wait_for_server, TestResult
        print("✅ Imported test utilities")
    except ImportError as e:
        print(f"❌ Error importing test_utils: {e}")
        return None
    
    # Try to import each test module individually
    try:
        # Fix the typo here:
        from test_basic_api import test_basic_endpoints, test_provider_endpoints, test_vm_endpoints
        test_modules['basic_api'] = [test_basic_endpoints, test_provider_endpoints, test_vm_endpoints]
        print("✅ Imported basic API tests")
    except ImportError as e:
        print(f"⚠️  Could not import basic API tests: {e}")
        test_modules['basic_api'] = []
    
    try:
        from test_workloads import test_workload_endpoints
        test_modules['workloads'] = [test_workload_endpoints]
        print("✅ Imported workload tests")
    except ImportError as e:
        print(f"⚠️  Could not import workload tests: {e}")
        test_modules['workloads'] = []
    
    try:
        from test_simulation import test_simulation_endpoints
        test_modules['simulation'] = [test_simulation_endpoints]
        print("✅ Imported simulation tests")
    except ImportError as e:
        print(f"⚠️  Could not import simulation tests: {e}")
        test_modules['simulation'] = []
    
    # ML tests are optional
    try:
        from test_ml import test_ml_endpoints
        test_modules['ml'] = [test_ml_endpoints]
        print("✅ Imported ML tests")
    except ImportError as e:
        print(f"⚠️  Could not import ML tests: {e}")
        test_modules['ml'] = []
    
    # Performance tests are optional
    try:
        from test_performance import test_concurrent_requests
        test_modules['performance'] = [test_concurrent_requests]
        print("✅ Imported performance tests")
    except ImportError as e:
        print(f"⚠️  Could not import performance tests: {e}")
        test_modules['performance'] = []
    
    return test_modules

def run_available_tests():
    """Run whatever tests are available"""
    from test_utils import wait_for_server, APITestClient
    
    print("🚀 AI Cloud Scheduler Backend Test Suite")
    print("="*60)
    
    # Check if server is running
    if not wait_for_server():
        print("❌ API server is not running!")
        print("Please start the server with: python api.py")
        return False
    
    client = APITestClient()
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic connectivity
    total_tests += 1
    try:
        response = client.get("/")
        if response.status_code == 200:
            print("✅ Basic API connectivity")
            tests_passed += 1
        else:
            print("❌ Basic API connectivity failed")
    except Exception as e:
        print(f"❌ Basic API connectivity error: {e}")
    
    # Test 2: Health check
    total_tests += 1
    try:
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health check endpoint")
            tests_passed += 1
        else:
            print("❌ Health check endpoint failed")
    except Exception as e:
        print(f"❌ Health check endpoint error: {e}")
    
    # Test 3: Providers
    total_tests += 1
    try:
        response = client.get("/api/providers/default")
        if response.status_code == 200 and len(response.json()) >= 3:
            print("✅ Provider endpoints")
            tests_passed += 1
        else:
            print("❌ Provider endpoints failed")
    except Exception as e:
        print(f"❌ Provider endpoints error: {e}")
    
    # Test 4: VMs
    total_tests += 1
    try:
        response = client.get("/api/vms/default")
        if response.status_code == 200 and len(response.json()) >= 4:
            print("✅ VM endpoints")
            tests_passed += 1
        else:
            print("❌ VM endpoints failed")
    except Exception as e:
        print(f"❌ VM endpoints error: {e}")
    
    # Test 5: Workloads
    total_tests += 1
    try:
        response = client.get("/api/workloads/sample")
        if response.status_code == 200 and len(response.json()) >= 8:
            print("✅ Workload endpoints")
            tests_passed += 1
        else:
            print("❌ Workload endpoints failed")
    except Exception as e:
        print(f"❌ Workload endpoints error: {e}")
    
    # Test 6: Basic simulation
    total_tests += 1
    try:
        workloads_response = client.get("/api/workloads/sample")
        if workloads_response.status_code == 200:
            workloads = workloads_response.json()[:3]
            simulation_data = {
                "scheduler_type": "random",
                "workloads": workloads
            }
            response = client.post("/api/simulation/run", json=simulation_data, timeout=30)
            if response.status_code == 200:
                print("✅ Basic simulation")
                tests_passed += 1
            else:
                print("❌ Basic simulation failed")
        else:
            print("❌ Basic simulation - cannot get workloads")
    except Exception as e:
        print(f"❌ Basic simulation error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🏁 TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {total_tests - tests_passed}")
    
    if total_tests > 0:
        success_rate = (tests_passed / total_tests) * 100
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        if tests_passed == total_tests:
            print(f"🎉 ALL TESTS PASSED! Your backend is working!")
        elif tests_passed >= total_tests * 0.8:
            print(f"🎯 Most tests passed! Minor issues detected.")
        else:
            print(f"⚠️  Several issues detected. Check your API implementation.")
    
    return tests_passed == total_tests

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run AI Cloud Scheduler Backend Tests')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML tests')
    parser.add_argument('--quick', action='store_true', help='Run basic tests only')
    
    args = parser.parse_args()
    
    # Try to import test modules
    test_modules = import_test_modules()
    
    if test_modules is None:
        print("❌ Cannot import test utilities. Running basic tests instead.")
        return run_available_tests()
    
    # Check if we have any test modules
    available_modules = [name for name, funcs in test_modules.items() if funcs]
    
    if not available_modules:
        print("⚠️  No test modules available. Running basic tests.")
        return run_available_tests()
    
    # Import and run whatever tests are available
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ API server is not running!")
        return False
    
    print(f"🚀 Running tests for available modules: {', '.join(available_modules)}")
    
    all_results = []
    
    for module_name, test_functions in test_modules.items():
        if not test_functions:
            continue
            
        if args.skip_ml and module_name == 'ml':
            continue
            
        print(f"\n{'='*20} {module_name.upper()} TESTS {'='*20}")
        
        for test_func in test_functions:
            try:
                result = test_func()
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error running {test_func.__name__}: {e}")
    
    # Summary
    if all_results:
        total_passed = sum(r.passed for r in all_results)
        total_failed = sum(r.failed for r in all_results)
        
        print(f"\n{'='*60}")
        print(f"🏁 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total Tests: {total_passed + total_failed}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        
        return total_failed == 0
    else:
        return run_available_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)