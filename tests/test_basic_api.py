import requests
import json
from test_utils import APITestClient, TestResult

def test_basic_endpoints():  # Fixed typo: was test_basic_endpoinnts
    """Test basic API endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("🧪 Testing Basic API Endpoints")
    print("-" * 40)
    
    # Test 1: Root endpoint
    try:
        response = client.get("/")
        if response.status_code == 200:
            data = response.json()
            if "message" in data and "version" in data:
                result.add_pass("Root endpoint (/)")
            else:
                result.add_fail("Root endpoint (/)", "Missing required fields in response")
        else:
            result.add_fail("Root endpoint (/)", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Root endpoint (/)", str(e))
    
    # Test 2: Health check
    try:
        response = client.get("/health")
        if response.status_code == 200:
            result.add_pass("Health check (/health)")
        else:
            result.add_fail("Health check (/health)", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Health check (/health)", str(e))
    
    # Test 3: API docs
    try:
        response = client.get("/docs")
        if response.status_code == 200:
            result.add_pass("API documentation (/docs)")
        else:
            result.add_fail("API documentation (/docs)", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("API documentation (/docs)", str(e))
    
    # Test 4: OpenAPI schema
    try:
        response = client.get("/openapi.json")
        if response.status_code == 200:
            data = response.json()
            if "openapi" in data and "info" in data:
                result.add_pass("OpenAPI schema (/openapi.json)")
            else:
                result.add_fail("OpenAPI schema (/openapi.json)", "Invalid OpenAPI schema")
        else:
            result.add_fail("OpenAPI schema (/openapi.json)", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("OpenAPI schema (/openapi.json)", str(e))
    
    return result

def test_provider_endpoints():
    """Test cloud provider endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("\n🧪 Testing Provider Endpoints")
    print("-" * 40)
    
    # Test 1: Get default providers
    try:
        response = client.get("/api/providers/default")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 3:
                # Check if we have AWS, GCP, Azure
                provider_names = [p.get("name", "").lower() for p in data]
                if "aws" in provider_names and "gcp" in provider_names and "azure" in provider_names:
                    result.add_pass("Get default providers")
                else:
                    result.add_fail("Get default providers", "Missing expected providers (AWS, GCP, Azure)")
            else:
                result.add_fail("Get default providers", "Invalid response format")
        else:
            result.add_fail("Get default providers", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get default providers", str(e))
    
    # Test 2: Create custom provider
    try:
        provider_data = {
            "name": "TestProvider",
            "cpu_cost": 0.05,
            "memory_cost_gb": 0.012
        }
        response = client.post("/api/providers", json=provider_data)
        if response.status_code == 200:
            data = response.json()
            if data.get("name") == "TestProvider":
                result.add_pass("Create custom provider")
            else:
                result.add_fail("Create custom provider", "Provider not created correctly")
        else:
            result.add_fail("Create custom provider", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Create custom provider", str(e))
    
    return result

def test_vm_endpoints():
    """Test virtual machine endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("\n🧪 Testing VM Endpoints")
    print("-" * 40)
    
    # Test 1: Get default VMs
    try:
        response = client.get("/api/vms/default")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 4:
                # Check VM structure
                vm = data[0]
                required_fields = ["vm_id", "cpu_capacity", "memory_capacity_gb", "provider"]
                if all(field in vm for field in required_fields):
                    result.add_pass("Get default VMs")
                else:
                    result.add_fail("Get default VMs", "Missing required VM fields")
            else:
                result.add_fail("Get default VMs", "Invalid response format")
        else:
            result.add_fail("Get default VMs", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get default VMs", str(e))
    
    # Test 2: Create custom VM
    try:
        vm_data = {
            "vm_id": 999,
            "cpu_capacity": 8,
            "memory_capacity_gb": 16,
            "provider": {
                "name": "TestProvider",
                "cpu_cost": 0.05,
                "memory_cost_gb": 0.012
            }
        }
        response = client.post("/api/vms", json=vm_data)
        if response.status_code == 200:
            data = response.json()
            if data.get("vm_id") == 999:
                result.add_pass("Create custom VM")
            else:
                result.add_fail("Create custom VM", "VM not created correctly")
        else:
            result.add_fail("Create custom VM", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Create custom VM", str(e))
    
    return result

if __name__ == "__main__":
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ Cannot run tests - API server is not running")
        print("Please start the API server with: python api.py")
        exit(1)
    
    # Run all basic tests
    results = []
    results.append(test_basic_endpoints())
    results.append(test_provider_endpoints())
    results.append(test_vm_endpoints())
    
    # Combined summary
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    
    print(f"\n{'='*50}")
    print(f"BASIC API TESTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {(total_passed/(total_passed+total_failed))*100:.1f}%")