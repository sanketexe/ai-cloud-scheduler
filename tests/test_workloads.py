import os
from test_utils import APITestClient, TestResult, create_test_workload_csv

def test_workload_endpoints():
    """Test workload management endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("🧪 Testing Workload Endpoints")
    print("-" * 40)
    
    # Test 1: Get sample workloads
    try:
        response = client.get("/api/workloads/sample")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 8:
                # Check workload structure
                workload = data[0]
                required_fields = ["id", "cpu_required", "memory_required_gb"]
                if all(field in workload for field in required_fields):
                    result.add_pass("Get sample workloads")
                else:
                    result.add_fail("Get sample workloads", "Missing required workload fields")
            else:
                result.add_fail("Get sample workloads", "Invalid response format")
        else:
            result.add_fail("Get sample workloads", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get sample workloads", str(e))
    
    # Test 2: Create workload
    try:
        workload_data = {
            "id": 999,
            "cpu_required": 4,
            "memory_required_gb": 8
        }
        response = client.post("/api/workloads", json=workload_data)
        if response.status_code == 200:
            data = response.json()
            if data.get("id") == 999:
                result.add_pass("Create workload")
            else:
                result.add_fail("Create workload", "Workload not created correctly")
        else:
            result.add_fail("Create workload", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Create workload", str(e))
    
    # Test 3: Generate random workloads
    try:
        response = client.post("/api/workloads/generate", params={"count": 5})
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) == 5:
                result.add_pass("Generate random workloads")
            else:
                result.add_fail("Generate random workloads", "Invalid number of workloads generated")
        else:
            result.add_fail("Generate random workloads", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Generate random workloads", str(e))
    
    # Test 4: Upload workloads CSV
    try:
        csv_file = create_test_workload_csv()
        with open(csv_file, 'rb') as f:
            files = {'file': ('test_workloads.csv', f, 'text/csv')}
            response = client.post("/api/workloads/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if "workloads" in data and len(data["workloads"]) == 8:
                result.add_pass("Upload workloads CSV")
            else:
                result.add_fail("Upload workloads CSV", "Incorrect number of workloads uploaded")
        else:
            result.add_fail("Upload workloads CSV", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Upload workloads CSV", str(e))
    
    # Test 5: Invalid workload data
    try:
        invalid_data = {
            "id": "invalid",  # Should be integer
            "cpu_required": -1,  # Should be positive
            "memory_required_gb": "invalid"  # Should be number
        }
        response = client.post("/api/workloads", json=invalid_data)
        if response.status_code == 422:  # Validation error expected
            result.add_pass("Reject invalid workload data")
        else:
            result.add_fail("Reject invalid workload data", "Should have rejected invalid data")
    except Exception as e:
        result.add_fail("Reject invalid workload data", str(e))
    
    return result

if __name__ == "__main__":
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ Cannot run tests - API server is not running")
        exit(1)
    
    result = test_workload_endpoints()
    result.summary()