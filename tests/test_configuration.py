"""
Configuration management tests for AI Cloud Scheduler
"""

import requests
import json
from test_utils import APITestClient, TestResult

def test_configuration_endpoints():
    """Test configuration management endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("🧪 Testing Configuration Management")
    print("-" * 40)
    
    # Test 1: Get all configurations
    try:
        response = client.get("/api/config")
        if response.status_code == 200:
            data = response.json()
            if "configurations" in data and "categories" in data:
                categories = data["categories"]
                expected_categories = ["api", "scheduler", "ml", "providers", "performance"]
                if all(cat in categories for cat in expected_categories):
                    result.add_pass("Get all configurations")
                else:
                    result.add_fail("Get all configurations", "Missing expected categories")
            else:
                result.add_fail("Get all configurations", "Invalid response format")
        else:
            result.add_fail("Get all configurations", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get all configurations", str(e))
    
    # Test 2: Show configuration overview
    try:
        response = client.get("/api/config/show")
        if response.status_code == 200:
            data = response.json()
            if "system_overview" in data and "categories" in data:
                system_overview = data["system_overview"]
                if "total_categories" in system_overview and "status" in system_overview:
                    result.add_pass("Show configuration overview")
                else:
                    result.add_fail("Show configuration overview", "Missing system overview fields")
            else:
                result.add_fail("Show configuration overview", "Invalid response format")
        else:
            result.add_fail("Show configuration overview", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Show configuration overview", str(e))
    
    # Test 3: Get specific configuration (API category)
    try:
        response = client.get("/api/config/api")
        if response.status_code == 200:
            data = response.json()
            if data["category"] == "api" and "config" in data:
                config = data["config"]
                if "version" in config and "port" in config:
                    result.add_pass("Get API configuration")
                else:
                    result.add_fail("Get API configuration", "Missing API config fields")
            else:
                result.add_fail("Get API configuration", "Invalid response format")
        else:
            result.add_fail("Get API configuration", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get API configuration", str(e))
    
    # Test 4: Get scheduler configuration
    try:
        response = client.get("/api/config/scheduler")
        if response.status_code == 200:
            data = response.json()
            if data["category"] == "scheduler" and "config" in data:
                config = data["config"]
                if "default_algorithm" in config and "available_algorithms" in config:
                    algorithms = config["available_algorithms"]
                    expected_algorithms = ["random", "lowest_cost", "round_robin"]
                    if all(alg in algorithms for alg in expected_algorithms):
                        result.add_pass("Get scheduler configuration")
                    else:
                        result.add_fail("Get scheduler configuration", "Missing scheduling algorithms")
                else:
                    result.add_fail("Get scheduler configuration", "Missing scheduler config fields")
            else:
                result.add_fail("Get scheduler configuration", "Invalid response format")
        else:
            result.add_fail("Get scheduler configuration", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get scheduler configuration", str(e))
    
    # Test 5: Update configuration
    try:
        update_data = {
            "category": "api",
            "config": {
                "debug": True,
                "max_workers": 8
            }
        }
        response = client.post("/api/config/api", data=update_data)
        if response.status_code == 200:
            data = response.json()
            if "message" in data and "updated_config" in data:
                updated_config = data["updated_config"]
                if updated_config.get("debug") == True and updated_config.get("max_workers") == 8:
                    result.add_pass("Update configuration")
                else:
                    result.add_fail("Update configuration", "Configuration not updated correctly")
            else:
                result.add_fail("Update configuration", "Invalid response format")
        else:
            result.add_fail("Update configuration", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Update configuration", str(e))
    
    # Test 6: Export configuration
    try:
        response = client.get("/api/config/export")
        if response.status_code == 200:
            data = response.json()
            if "export_info" in data and "configurations" in data:
                export_info = data["export_info"]
                if "timestamp" in export_info and "version" in export_info:
                    result.add_pass("Export configuration")
                else:
                    result.add_fail("Export configuration", "Missing export info fields")
            else:
                result.add_fail("Export configuration", "Invalid response format")
        else:
            result.add_fail("Export configuration", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Export configuration", str(e))
    
    # Test 7: Get invalid configuration category
    try:
        response = client.get("/api/config/invalid_category")
        if response.status_code == 404:
            result.add_pass("Handle invalid configuration category")
        else:
            result.add_fail("Handle invalid configuration category", f"Expected 404, got {response.status_code}")
    except Exception as e:
        result.add_fail("Handle invalid configuration category", str(e))
    
    # Test 8: Provider configurations
    try:
        response = client.get("/api/config/providers")
        if response.status_code == 200:
            data = response.json()
            if data["category"] == "providers" and "config" in data:
                config = data["config"]
                providers = ["aws", "gcp", "azure"]
                if all(provider in config for provider in providers):
                    # Check if providers have required fields
                    aws_config = config.get("aws", {})
                    if "enabled" in aws_config and "cpu_cost" in aws_config:
                        result.add_pass("Get provider configurations")
                    else:
                        result.add_fail("Get provider configurations", "Missing provider config fields")
                else:
                    result.add_fail("Get provider configurations", "Missing cloud providers")
            else:
                result.add_fail("Get provider configurations", "Invalid response format")
        else:
            result.add_fail("Get provider configurations", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get provider configurations", str(e))
    
    return result

def test_configuration_validation():
    """Test configuration validation and error handling"""
    client = APITestClient()
    result = TestResult()
    
    print("\n🧪 Testing Configuration Validation")
    print("-" * 40)
    
    # Test 1: Invalid update data
    try:
        invalid_data = {
            "category": "api",
            "config": "not_a_dict"  # Should be a dictionary
        }
        response = client.post("/api/config/api", data=invalid_data)
        if response.status_code == 422:  # Validation error
            result.add_pass("Reject invalid configuration data")
        else:
            result.add_fail("Reject invalid configuration data", f"Expected 422, got {response.status_code}")
    except Exception as e:
        result.add_fail("Reject invalid configuration data", str(e))
    
    # Test 2: Category mismatch
    try:
        mismatch_data = {
            "category": "scheduler",  # Different from URL
            "config": {"debug": True}
        }
        response = client.post("/api/config/api", data=mismatch_data)  # URL says 'api'
        if response.status_code == 400:
            result.add_pass("Detect category mismatch")
        else:
            result.add_fail("Detect category mismatch", f"Expected 400, got {response.status_code}")
    except Exception as e:
        result.add_fail("Detect category mismatch", str(e))
    
    return result

if __name__ == "__main__":
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ Cannot run tests - API server is not running")
        print("Please start the API server with: python api.py")
        exit(1)
    
    # Run all configuration tests
    results = []
    results.append(test_configuration_endpoints())
    results.append(test_configuration_validation())
    
    # Combined summary
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    
    print(f"\n{'='*50}")
    print(f"CONFIGURATION TESTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    if total_passed + total_failed > 0:
        success_rate = (total_passed / (total_passed + total_failed)) * 100
        print(f"📊 Success Rate: {success_rate:.1f}%")
    
    if total_failed == 0:
        print("🎉 All configuration tests passed!")
    else:
        print("⚠️ Some configuration tests failed")
        for result in results:
            for error in result.errors:
                print(f"   - {error}")