#!/usr/bin/env python3
"""
Quick smoke tests for AI Cloud Scheduler
"""

import sys
import os
import requests
import time

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 10

def wait_for_server(max_attempts=10, delay=1):
    """Wait for API server to be ready"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("✅ API server is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            print(f"⏳ Waiting for API server... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
    
    print("❌ API server is not responding")
    return False

def quick_smoke_test():
    """Run essential smoke tests"""
    print("🔥 Quick Smoke Test for AI Cloud Scheduler")
    print("-" * 50)
    
    if not wait_for_server():
        print("❌ API server is not running!")
        print("Start server with: python api.py")
        return False
    
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Basic connectivity
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            print("✅ 1/7 - API server responding")
            tests_passed += 1
        else:
            print("❌ 1/7 - API server not responding correctly")
    except Exception as e:
        print(f"❌ 1/7 - Cannot connect to API server: {e}")
    
    # Test 2: API Documentation
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            print("✅ 2/7 - API documentation accessible")
            tests_passed += 1
        else:
            print("❌ 2/7 - API documentation failed")
    except Exception as e:
        print(f"❌ 2/7 - API documentation error: {e}")
    
    # Test 3: Get providers
    try:
        response = requests.get(f"{API_BASE_URL}/api/providers/default", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 3:
                provider_names = [p.get("name", "").lower() for p in data]
                if "aws" in provider_names and "gcp" in provider_names and "azure" in provider_names:
                    print("✅ 3/7 - Provider endpoints working")
                    tests_passed += 1
                else:
                    print("❌ 3/7 - Provider endpoints: missing expected providers")
            else:
                print(f"❌ 3/7 - Provider endpoints: invalid response format")
        else:
            print(f"❌ 3/7 - Provider endpoints failed: {response.status_code}")
    except Exception as e:
        print(f"❌ 3/7 - Provider endpoints error: {e}")
    
    # Test 4: Get VMs
    try:
        response = requests.get(f"{API_BASE_URL}/api/vms/default", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 4:
                # Check if VMs have required fields
                vm = data[0] if data else {}
                required_fields = ["vm_id", "cpu_capacity", "memory_capacity_gb", "provider"]
                if all(field in vm for field in required_fields):
                    print("✅ 4/7 - VM endpoints working")
                    tests_passed += 1
                else:
                    print("❌ 4/7 - VM endpoints: missing required fields")
            else:
                print(f"❌ 4/7 - VM endpoints: invalid response format")
        else:
            print(f"❌ 4/7 - VM endpoints failed: {response.status_code}")
    except Exception as e:
        print(f"❌ 4/7 - VM endpoints error: {e}")
    
    # Test 5: Get workloads
    try:
        response = requests.get(f"{API_BASE_URL}/api/workloads/sample", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 8:
                print("✅ 5/7 - Workload endpoints working")
                tests_passed += 1
            else:
                print(f"❌ 5/7 - Workload endpoints: insufficient workloads ({len(data)} < 8)")
        else:
            print(f"❌ 5/7 - Workload endpoints failed: {response.status_code}")
    except Exception as e:
        print(f"❌ 5/7 - Workload endpoints error: {e}")
    
    # Test 6: Quick simulation
    try:
        workloads_response = requests.get(f"{API_BASE_URL}/api/workloads/sample", timeout=REQUEST_TIMEOUT)
        if workloads_response.status_code == 200:
            workloads = workloads_response.json()[:3]  # Use first 3 workloads
            simulation_data = {
                "scheduler_type": "random",
                "workloads": workloads
            }
            response = requests.post(f"{API_BASE_URL}/api/simulation/run", 
                                   json=simulation_data, timeout=30)
            if response.status_code == 200:
                print("✅ 6/7 - Basic simulation working")
                tests_passed += 1
            else:
                print(f"❌ 6/7 - Basic simulation failed: {response.status_code}")
        else:
            print("❌ 6/7 - Cannot get workloads for simulation")
    except Exception as e:
        print(f"❌ 6/7 - Simulation error: {e}")
    
    # Test 7: Configuration system
    try:
        response = requests.get(f"{API_BASE_URL}/api/config/show", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "system_overview" in data and "categories" in data:
                total_categories = data["system_overview"].get("total_categories", 0)
                if total_categories >= 7:  # Should have at least 7 config categories
                    print("✅ 7/7 - Configuration system working")
                    tests_passed += 1
                else:
                    print(f"❌ 7/7 - Configuration system: insufficient categories ({total_categories} < 7)")
            else:
                print("❌ 7/7 - Configuration system: invalid response format")
        else:
            print(f"❌ 7/7 - Configuration system failed: {response.status_code}")
    except Exception as e:
        print(f"❌ 7/7 - Configuration system error: {e}")
    
    # Summary
    print(f"\n🏁 Quick Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All essential features are working!")
        print("✨ Your backend is ready for use!")
        return True
    elif tests_passed >= 4:
        print("⚠️  Most features working, some issues detected")
        print("💡 Run 'python run_tests.py' for detailed testing")
        return True
    else:
        print("❌ Major issues detected")
        print("💡 Check your API implementation")
        return False

if __name__ == "__main__":
    success = quick_smoke_test()
    print(f"\n📋 Next steps:")
    print(f"   - For detailed testing: python run_tests.py")
    print(f"   - For basic validation: python run_tests.py --quick")
    sys.exit(0 if success else 1)